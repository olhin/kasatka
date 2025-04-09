import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import warnings
from sklearn.preprocessing import LabelEncoder
import time
import threading
import socket
import struct
import asyncio
import websockets
import json
import argparse

# Импорт TensorFlow вместо tflite_runtime
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("ВНИМАНИЕ: TensorFlow не установлен. Демо-режим не будет работать.")
    print("Для установки TensorFlow выполните: pip install tensorflow")

def find_model_file(filename):
    """Ищет файл модели в нескольких возможных местах"""
    possible_paths = [
        filename,                     # Текущая директория
        f"../{filename}",             # Родительская директория
        f"../../{filename}",          # Директория на два уровня выше
        f"../../../{filename}",       # Директория на три уровня выше
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Найден файл: {path}")
            return path
    
    print(f"Не удалось найти файл {filename} ни в одном из следующих мест:")
    for path in possible_paths:
        print(f"  - {path}")
    return None

# Конфигурация
PORTS = [5000, 5001, 5002, 5003]
TARGET_SAMPLE_RATE = 44100
N_MFCC = 40
MODEL_PATH = 'sound_classifier_model.pth'
TFLITE_MODEL_PATH = 'soundclassifier_with_metadata.tflite'
LABELS_PATH = 'labels.txt'
BUFFER_SIZE = 44100 * 2  # 2 секунды аудио
MAX_PACKET_SIZE = 4096
DROPOUT_RATE = 0.6
DATA_DIRS = {
    'train': 'train',
    'valid': 'valid',
    'test': 'test'
}
WEBSOCKET_PORT = 8765  # Порт для WebSocket сервера
DEBUG_MODE = False  # Включение/отключение отладочного вывода

# Глобальная переменная для хранения активных WebSocket соединений
connected_clients = set()

# Режим работы (дрон или хлопок)
OPERATING_MODE = "drone"  # По умолчанию режим дрона

# Функция для отправки данных через WebSocket
async def send_to_clients(data):
    if connected_clients:  # Проверка наличия подключенных клиентов
        # Создаем JSON-объект для отправки
        message = json.dumps(data)
        # Отправляем всем подключенным клиентам
        try:
            await asyncio.gather(
                *[client.send(message) for client in connected_clients],
                return_exceptions=True
            )
            print(f"WebSocket: отправлены данные о секторе '{data['sector']}' {len(connected_clients)} клиентам")
        except Exception as e:
            print(f"Ошибка WebSocket при отправке: {str(e)}")
    else:
        print("WebSocket: нет подключенных клиентов")
    return True  # Возвращаем True для подтверждения отправки

# Обработчик WebSocket соединений
async def websocket_handler(websocket, path):
    # Регистрируем нового клиента
    connected_clients.add(websocket)
    print(f"Новое WebSocket соединение: {len(connected_clients)} активных соединений")
    try:
        # Отправляем приветственное сообщение для проверки соединения
        await websocket.send(json.dumps({"status": "connected", "message": "Соединение установлено"}))
        # Держим соединение открытым
        await websocket.wait_closed()
    finally:
        # Удаляем клиента при отключении
        connected_clients.remove(websocket)
        print(f"WebSocket соединение закрыто: {len(connected_clients)} активных соединений")

# Запуск WebSocket сервера
async def start_websocket_server():
    print(f"Запуск WebSocket сервера на порту {WEBSOCKET_PORT}...")
    async with websockets.serve(websocket_handler, "0.0.0.0", WEBSOCKET_PORT):
        await asyncio.Future()  # Бесконечное ожидание

class SoundClassifier:
    def __init__(self, mode="drone"):
        global OPERATING_MODE
        OPERATING_MODE = mode
        
        self.le = LabelEncoder()
        self.model = None
        self.running = True
        self.lock = threading.Lock()
        self.audio_buffers = {port: np.array([], dtype=np.float32) for port in PORTS}
        self.last_predictions = {}
        self.sample_rates = {}
        self.sockets = {}  # Для хранения сокетов
        
        # Добавляем таймер для отслеживания времени последнего обнаружения
        self.last_detection_time = 0
        self.current_active_sector = "Не определен"
        
        # Инициализация путей к данным
        self.data_dir = DATA_DIRS['train']
        self.valid_dir = DATA_DIRS['valid']
        self.test_dir = DATA_DIRS['test']
        
        print(f"Инициализация в режиме: {OPERATING_MODE}")
        
        if OPERATING_MODE == "drone":
            if not self.load_model():  # Попытка загрузки модели
                self.init_training()    # Запуск обучения если модель не найдена
        else:  # Режим хлопка
            # Проверяем, доступен ли TensorFlow
            if not TENSORFLOW_AVAILABLE:
                print("Критическая ошибка: TensorFlow не установлен, но выбран режим хлопка!")
                print("Пожалуйста, установите TensorFlow: pip install tensorflow")
                print("Или запустите install_dependencies.bat")
                print("Переключение на стандартный режим дрона...")
                OPERATING_MODE = "drone"
                if not self.load_model():  # Попытка загрузки модели
                    self.init_training()   # Запуск обучения если модель не найдена
            elif not self.load_tflite_model():
                print("Критическая ошибка: не удалось загрузить TensorFlow модель для режима хлопка!")
                print("Переключение на стандартный режим дрона...")
                OPERATING_MODE = "drone"
                if not self.load_model():  # Попытка загрузки модели
                    self.init_training()   # Запуск обучения если модель не найдена
            
        self.init_network()

    def load_tflite_model(self):
        """Загрузка TensorFlow модели для режима хлопка"""
        try:
            # Проверяем наличие TensorFlow
            if not TENSORFLOW_AVAILABLE:
                print("Ошибка: TensorFlow не установлен. Невозможно использовать демо-режим.")
                return False
                
            # Поиск файла модели
            tflite_path = find_model_file(TFLITE_MODEL_PATH)
            if not tflite_path:
                print(f"Файл TFLite модели не найден.")
                return False
                
            # Поиск файла меток
            labels_path = find_model_file(LABELS_PATH)
            if not labels_path:
                print(f"Файл меток не найден.")
                labels_path = None  # Используем метки по умолчанию
                
            # Загрузка модели TensorFlow Lite с помощью интерпретатора
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()
            
            # Получение информации о входных и выходных тензорах
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Загрузка меток классов
            if labels_path and os.path.exists(labels_path):
                with open(labels_path, 'r', encoding='utf-8') as f:
                    self.labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
                print(f"Загружены метки классов: {self.labels}")
            else:
                print(f"Используются метки по умолчанию.")
                self.labels = ["Class 2", "Фоновый шум"]  # Метки по умолчанию
                
            print(f"TensorFlow Lite модель успешно загружена")
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке TensorFlow модели: {str(e)}")
            return False

    # 🔄 Существующий метод save_model
    def save_model(self):
        """Сохранение модели на диск"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'le_classes': self.le.classes_,
                'sample_rates': self.sample_rates,
                'input_size': N_MFCC,
                'num_classes': len(self.le.classes_)
            }, MODEL_PATH)
            print(f"Модель сохранена в {MODEL_PATH}")
        except Exception as e:
            print(f"Ошибка сохранения: {str(e)}")
            raise

    def load_model(self):
        """Загрузка модели с диска"""
        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(
                    MODEL_PATH,
                    map_location='cpu',
                    weights_only=False
                )
                self.le.classes_ = checkpoint['le_classes']
                self.sample_rates = checkpoint.get('sample_rates', {})
                
                # Создаем модель с правильной архитектурой
                input_size = checkpoint['input_size']
                hidden_size = 256  # Используем 256 нейронов в скрытом слое
                num_classes = checkpoint['num_classes']
                
                self.model = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT_RATE),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.BatchNorm1d(hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT_RATE),
                    nn.Linear(hidden_size // 2, num_classes)
                )
                
                # Загружаем веса
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Модель загружена из {MODEL_PATH}")
                return True
            except Exception as e:
                print(f"Ошибка загрузки модели: {str(e)}")
                return False
        print("Файл модели не найден")
        return False

    def init_training(self):
        """Инициализация процесса обучения при отсутствии модели"""
        print("\n" + "="*60)
        print("Предупреждение: Модель не найдена! Запуск обучения...")
        print("="*60 + "\n")
        
        try:
            self.load_data()
            self.create_model()
            self.train(num_epochs=3000)  # Увеличиваем количество эпох до 3000
            self.save_model()
            print("\nОбучение успешно завершено!\n")
            self.load_model()
        except Exception as e:
            print(f"Критическая ошибка при обучении: {str(e)}")
            exit(1)

    def load_data(self):
        def load_from_dir(dir_path):
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Директория {dir_path} не найдена")
                
            X, y = [], []
            file_count = 0
            start_time = time.time()
            
            try:
                for label in os.listdir(dir_path):
                    if not self.running:
                        raise KeyboardInterrupt
                        
                    label_dir = os.path.join(dir_path, label)
                    if not os.path.isdir(label_dir):
                        print(f"Пропускаем {label_dir} - не директория")
                        continue
                        
                    print(f"Обработка класса: {label}")
                    files = [f for f in os.listdir(label_dir) if f.endswith('.wav')]
                    if not files:
                        print(f"Внимание: нет .wav файлов в {label_dir}")
                        continue
                        
                    for file in files:
                        try:
                            if not self.running:
                                raise KeyboardInterrupt
                                
                            file_path = os.path.join(label_dir, file)
                            file_count += 1
                            
                            audio, sr = librosa.load(file_path, sr=None, mono=True)
                            if sr != TARGET_SAMPLE_RATE:
                                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
                                
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                mfcc = librosa.feature.mfcc(
                                    y=audio,
                                    sr=TARGET_SAMPLE_RATE,
                                    n_mfcc=N_MFCC,
                                    n_fft=2048,
                                    hop_length=512
                                )
                                
                            X.append(np.mean(mfcc.T, axis=0))
                            y.append(label)
                            
                            if file_count % 10 == 0:
                                elapsed = time.time() - start_time
                                print(f"Обработано {file_count} файлов ({elapsed:.1f} сек)")
                                
                        except KeyboardInterrupt:
                            print("\nПрерывание загрузки данных...")
                            self.running = False
                            raise
                            
                        except Exception as e:
                            print(f"Ошибка обработки {file_path}: {str(e)}")
                            continue
                            
            except KeyboardInterrupt:
                print("\nЗагрузка данных прервана пользователем")
                self.running = False
                raise
                
            print(f"Успешно загружено {file_count} файлов")
            return np.array(X, dtype=np.float32), np.array(y)

        try:
            for path in [self.data_dir, self.valid_dir, self.test_dir]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Директория {path} не существует")

            print("\nЗагрузка тренировочных данных...")
            self.X, self.y = load_from_dir(self.data_dir)
            self.y_encoded = self.le.fit_transform(self.y)
            
            print("\nЗагрузка валидационных данных...")
            self.X_valid, self.y_valid = load_from_dir(self.valid_dir)
            self.y_encoded_valid = self.le.transform(self.y_valid)
            
            print("\nЗагрузка тестовых данных...")
            self.X_test, self.y_test = load_from_dir(self.test_dir)
            self.y_encoded_test = self.le.transform(self.y_test)
            
            print("\nСтатистика датасета:")
            self._print_dataset_stats("Тренировочные", self.y)
            self._print_dataset_stats("Валидационные", self.y_valid)
            self._print_dataset_stats("Тестовые", self.y_test)
            
        except KeyboardInterrupt:
            print("\nПолное прерывание загрузки данных")
            self.running = False
            raise
            
        except Exception as e:
            print(f"Критическая ошибка при загрузке данных: {str(e)}")
            self.running = False
            raise

    def _print_dataset_stats(self, name, labels):
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{name} данные:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} примеров")
        print(f"Всего: {len(labels)} примеров\n")

    def create_model(self):
        """Создание модели нейронной сети"""
        input_size = 40  # Размер входного вектора признаков (MFCC)
        hidden_size = 256  # Размер скрытого слоя
        num_classes = 2  # Количество классов (дрон/фон)
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Инициализация весов
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, num_epochs=250):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_encoded, dtype=torch.long)
    
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        learning_rates = []
    
        try:
            for epoch in range(num_epochs):
                if not self.running:
                    raise KeyboardInterrupt
                
            # Обучение
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = self.criterion(outputs, y_tensor)
                loss.backward()
                self.optimizer.step()
            
            # Расчет точности тренировки
                _, preds = torch.max(outputs, 1)
                correct = (preds == y_tensor).sum().item()
                train_accuracy = correct / len(y_tensor) * 100
            
                train_losses.append(loss.item())
                train_accuracies.append(train_accuracy)
            
            # Валидация
                self.model.eval()
                with torch.no_grad():
                    valid_outputs = self.model(torch.tensor(self.X_valid, dtype=torch.float32))
                    valid_loss = self.criterion(valid_outputs, torch.tensor(self.y_encoded_valid, dtype=torch.long))
                
                # Расчет точности валидации
                    _, valid_preds = torch.max(valid_outputs, 1)
                    valid_correct = (valid_preds == torch.tensor(self.y_encoded_valid, dtype=torch.long)).sum().item()
                    valid_accuracy = valid_correct / len(self.y_encoded_valid) * 100
                
                    valid_losses.append(valid_loss.item())
                    valid_accuracies.append(valid_accuracy)
            
                learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
                print(f"Эпоха [{epoch+1}/{num_epochs}] | "
                    f"Потеря: {loss.item():.4f} | Валидация: {valid_loss.item():.4f} | "
                    f"Точность: {train_accuracy:.2f}% | Валидационная точность: {valid_accuracy:.2f}%")

        except KeyboardInterrupt:
            print("\nОбучение прервано пользователем")
            self.running = False
        
        finally:
            self.analyze_training(
                train_losses, 
                valid_losses,
                train_accuracies,
                valid_accuracies,
                learning_rates
            )


    def analyze_training(self, train_losses, valid_losses, train_accuracies, valid_accuracies, learning_rates):
        print("\n" + "="*60)
        print("Детальный анализ обучения:")
        print("="*60)
    
    # Анализ точности
        max_train_acc = max(train_accuracies)
        min_train_acc = min(train_accuracies)
        final_train_acc = train_accuracies[-1]
    
        max_valid_acc = max(valid_accuracies)
        min_valid_acc = min(valid_accuracies)
        final_valid_acc = valid_accuracies[-1]
    
        print(f"\nТренировочная точность:")
        print(f"  Начальная: {train_accuracies[0]:.2f}%")
        print(f"  Максимальная: {max_train_acc:.2f}% (эпоха {train_accuracies.index(max_train_acc)+1})")
        print(f"  Минимальная: {min_train_acc:.2f}% (эпоха {train_accuracies.index(min_train_acc)+1})")
        print(f"  Финальная: {final_train_acc:.2f}%")
    
        print(f"\nВалидационная точность:")
        print(f"  Начальная: {valid_accuracies[0]:.2f}%")
        print(f"  Максимальная: {max_valid_acc:.2f}% (эпоха {valid_accuracies.index(max_valid_acc)+1})")
        print(f"  Минимальная: {min_valid_acc:.2f}% (эпоха {valid_accuracies.index(min_valid_acc)+1})")
        print(f"  Финальная: {final_valid_acc:.2f}%")
    
    # График точности
        print("\nДинамика точности:")
        for epoch, (train_acc, valid_acc) in enumerate(zip(train_accuracies, valid_accuracies)):
            diff = valid_acc - train_acc
            status = "↑↑" if diff > 5 else "↑↓" if diff < -5 else "≈"
            print(f"Эпоха {epoch+1:2d}: Train {train_acc:6.2f}% | Valid {valid_acc:6.2f}% | Разница {diff:+5.2f}% {status}")
    
    # Рекомендации
        print("\nРекомендации:")
        if final_valid_acc < 60:
            print("  ▸ Низкая точность! Попробуйте:")
            print("    - Увеличить размер датасета")
            print("    - Добавить аугментацию аудио")
            print("    - Изменить архитектуру модели")
        elif final_valid_acc < 80:
            print("  ▸ Средняя точность. Возможные улучшения:")
            print("    - Настроить гиперпараметры (LR, слои)")
            print("    - Добавить регуляризацию")
        else:
            print("  ▸ Отличный результат! Модель готова к использованию")
    
        print("="*60 + "\n")

    def init_network(self):
        """Инициализация сетевого кода"""
        print("Инициализация сетевого компонента...")
        
        # Создаем UDP-сокеты для каждого порта
        for port in PORTS:
            print(f"  Открытие порта {port}...")
            
            # Создание UDP-сокета
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            sock.bind(('0.0.0.0', port))
            sock.setblocking(False)
            
            # Сохраняем сокет
            self.sockets[port] = sock
            print(f"  Порт {port} открыт")
            
        # Запуск WebSocket сервера
        self._start_websocket_loop()
        
        # Запуск сетевых потоков для прослушивания
        self.listeners = []
        for port in PORTS:
            thread = threading.Thread(
                target=self.network_listener,
                args=(port,),
                daemon=True
            )
            thread.start()
            self.listeners.append(thread)
            
        print("Сетевой компонент инициализирован")
        
        # Запуск основного цикла обработки в отдельном потоке
        self.process_thread = threading.Thread(
            target=self.process_audio,
            daemon=True
        )
        self.process_thread.start()

    def _run_websocket_server(self):
        """Запуск WebSocket сервера"""
        try:
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.run_until_complete(start_websocket_server())
        except Exception as e:
            print(f"Ошибка запуска WebSocket сервера: {str(e)}")
        finally:
            if hasattr(self, 'event_loop'):
                self.event_loop.close()

    def _start_websocket_loop(self):
        """Запуск WebSocket сервера в отдельном потоке"""
        self.websocket_thread = threading.Thread(target=self._run_websocket_server, daemon=True)
        self.websocket_thread.start()

    def network_listener(self, port):
        """Прослушивание сетевого порта"""
        if port not in self.sockets:
            print(f"Порт {port} не инициализирован")
            return
            
        sock = self.sockets[port]
        print(f"Слушаем порт {port}...")
        
        # Константа для Windows WSAEWOULDBLOCK
        WSAEWOULDBLOCK = 10035
        
        # Счетчик ошибок для предотвращения чрезмерного логирования
        error_count = 0
        last_log_time = time.time()
        
        while self.running:
            try:
                data, addr = sock.recvfrom(MAX_PACKET_SIZE)
                # Сбрасываем счетчик ошибок при успешном получении данных
                error_count = 0
                last_log_time = time.time()
                self.process_packet(port, data)
            except socket.error as e:
                # Обрабатываем ошибку WSAEWOULDBLOCK (нормальная для неблокирующих сокетов)
                error_code = e.args[0]
                if error_code == WSAEWOULDBLOCK:
                    # Полностью подавляем вывод о WSAEWOULDBLOCK - это нормально для неблокирующих сокетов
                    # Отображаем сообщение только в режиме отладки и не чаще раза в минуту
                    current_time = time.time()
                    if DEBUG_MODE and (current_time - last_log_time > 60):
                        print(f"Порт {port} ожидает данные (это нормально)")
                        last_log_time = current_time
                else:
                    # Это другая ошибка сокета - её логируем всегда
                    print(f"Ошибка сокета на порту {port}: {str(e)}")
                # Небольшая пауза для снижения нагрузки на CPU
                time.sleep(0.02)
            except Exception as e:
                print(f"Критическая ошибка на порту {port}: {str(e)}")
                time.sleep(0.1)  # Пауза перед повторной попыткой

    def process_packet(self, port, data):
        """Обработка сетевых пакетов"""
        try:
            if len(data) < 8:  # Минимальный размер пакета (timestamp + хотя бы один сэмпл)
                print(f"Порт {port}: Получен слишком короткий пакет ({len(data)} байт)")
                return
                
            timestamp = struct.unpack('d', data[:8])[0]
            audio_chunk = np.frombuffer(data[8:], dtype=np.float32)
            
            # Проверка на валидность аудиоданных
            if len(audio_chunk) == 0:
                print(f"Порт {port}: Получен пустой аудиочанк")
                return
                
            if np.isnan(audio_chunk).any() or np.isinf(audio_chunk).any():
                print(f"Порт {port}: Обнаружены некорректные значения в аудиоданных")
                return
                
            # Проверка уровня сигнала
            rms = np.sqrt(np.mean(audio_chunk**2))
            if rms < 1e-6:  # Слишком тихий сигнал
                print(f"Порт {port}: Сигнал слишком тихий (RMS: {rms})")
                return
            
            with self.lock:
                self.audio_buffers[port] = np.concatenate([
                    self.audio_buffers[port], 
                    audio_chunk
                ])[-BUFFER_SIZE:]
                
        except Exception as e:
            print(f"Ошибка обработки пакета на порту {port}: {str(e)}")

    def predict(self, audio):
        """Классификация аудиосигнала"""
        if OPERATING_MODE == "drone":
            return self._predict_drone(audio)
        else:
            return self._predict_clap(audio)
            
    def _predict_drone(self, audio):
        """Классификация звука дрона с использованием PyTorch модели"""
        # Проверка длины аудио
        if len(audio) < TARGET_SAMPLE_RATE:
            return "Фоновый шум", 0.0  # Недостаточно данных
            
        try:
            # Извлечение MFCC признаков
            mfccs = librosa.feature.mfcc(y=audio, sr=TARGET_SAMPLE_RATE, n_mfcc=N_MFCC)
            mfccs_processed = np.mean(mfccs.T, axis=0)
            
            # Преобразование в тензор PyTorch
            features = torch.tensor(mfccs_processed, dtype=torch.float32).unsqueeze(0)
            
            # Получение предсказания
            with torch.no_grad():
                self.model.eval()
                logits = self.model(features)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                max_prob, predicted_class = torch.max(probabilities, 1)
                
            # Получаем метку класса и вероятность
            class_name = self.le.classes_[predicted_class.item()]
            confidence = max_prob.item()
            
            return class_name, confidence
            
        except Exception as e:
            print(f"Ошибка предсказания (PyTorch): {str(e)}")
            return "Ошибка", 0.0
            
    def _predict_clap(self, audio):
        """Классификация звука хлопка с использованием TensorFlow Lite модели"""
        # Проверка длины аудио
        if len(audio) < TARGET_SAMPLE_RATE:
            return "Фоновый шум", 0.0  # Недостаточно данных
            
        try:
            # Вычисляем уровень громкости
            rms = np.sqrt(np.mean(audio**2))
            dBFS = 20 * np.log10(rms) if rms > 0 else -np.inf
            
            # Если звук достаточно громкий, считаем его хлопком
            # Устанавливаем порог в -21 dBFS согласно требованию
            if dBFS > -21:
                print(f"Обнаружен громкий звук: {dBFS:.1f} dBFS - распознаём как хлопок")
                return "Class 2", 1.0  # Обнаружили хлопок с максимальной уверенностью
            
            # Извлечение признаков для модели
            mfccs = librosa.feature.mfcc(y=audio, sr=TARGET_SAMPLE_RATE, n_mfcc=40)
            mfccs_processed = np.mean(mfccs.T, axis=0)
            
            # Подготовка входных данных для TFLite модели
            input_shape = self.input_details[0]['shape']
            input_data = np.expand_dims(mfccs_processed, axis=0).astype(np.float32)
            
            # Проверяем, совпадает ли форма входных данных с ожидаемой
            if input_data.shape != tuple(input_shape):
                # Если нет, изменяем размерность
                input_data = np.resize(input_data, input_shape)
            
            # Выполнение предсказания
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Получение результатов
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Получение класса с максимальной вероятностью
            predicted_idx = np.argmax(output_data[0])
            confidence = output_data[0][predicted_idx]
            
            # Получаем метку класса
            class_name = self.labels[predicted_idx] if predicted_idx < len(self.labels) else f"Класс {predicted_idx}"
            
            # Выводим отладочную информацию только если уровень выше -30 dBFS
            if dBFS > -30:
                print(f"Результат классификации: {class_name}, уверенность: {confidence:.1%}, уровень: {dBFS:.1f} dBFS")
            
            return class_name, float(confidence)
            
        except Exception as e:
            print(f"Ошибка предсказания (TFLite): {str(e)}")
            return "Ошибка", 0.0

    def process_audio(self):
        """Основной цикл обработки аудио"""
        print("Запуск обработки аудио...")
        
        # Вывод заголовка таблицы статуса
        print("\n" + "-"*82)
        print(f"{'Порт':<8} {'Устройство':<12} {'Статус':<20} {'Уверенность':<10} {'Уровень':<12} {'Буфер':<10}")
        print("-"*82)
        print("\n" * 4)  # Добавляем пустые строки для начального вывода
            
        # Последнее время отправки данных на фронтенд
        last_websocket_update = 0
            
        # Основной цикл обработки
        while self.running:
            try:
                # Проверяем наличие данных в буферах
                with self.lock:
                    all_empty = all(len(buf) == 0 for buf in self.audio_buffers.values())
                    if all_empty:
                        time.sleep(0.1)
                        continue
                
                # Создаем копии буферов для обработки
                current_buffers = {}
                
                with self.lock:
                    for port, buffer in self.audio_buffers.items():
                        current_buffers[port] = buffer.copy() if len(buffer) > 0 else np.array([], dtype=np.float32)
                
                # Обрабатываем каждый буфер отдельно
                predictions = {}
                
                for port, buffer in current_buffers.items():
                    if len(buffer) == 0:
                        # Если буфер пуст, используем предыдущее предсказание или "молчание"
                        predictions[port] = self.last_predictions.get(port, ("Фоновый шум", 0.0, -np.inf))
                        continue
                    
                    # Для каждого порта делаем предсказание
                    class_name, confidence = self.predict(buffer)
                    
                    # Вычисляем уровень звука в дБ
                    if len(buffer) > 0:
                        rms = np.sqrt(np.mean(buffer**2))
                        dBFS = 20 * np.log10(rms) if rms > 0 else -np.inf
                    else:
                        dBFS = -np.inf
                    
                    # Сохраняем предсказание и уровень
                    predictions[port] = (class_name, confidence, dBFS)
                    self.last_predictions[port] = predictions[port]
                
                # Определяем текущий активный сектор
                current_sector = self.determine_sector()
                
                # Отправляем информацию о секторе через WebSocket
                # Отправляем каждые 100 мс или при изменении сектора
                current_time = time.time()
                should_update = (current_time - last_websocket_update >= 0.1) or (current_sector != self.current_active_sector)
                
                if should_update and hasattr(self, 'event_loop'):
                    last_websocket_update = current_time
                    data_to_send = {
                        "sector": current_sector,
                        "timestamp": current_time
                    }
                    
                    try:
                        # Используем asyncio для отправки данных через WebSocket
                        future = asyncio.run_coroutine_threadsafe(
                            send_to_clients(data_to_send),
                            self.event_loop
                        )
                        # Дожидаемся выполнения с таймаутом 0.5 секунды
                        future.result(timeout=0.5)
                    except Exception as e:
                        print(f"Ошибка отправки данных через WebSocket: {str(e)}")
                        
                    # Выводим отладочную информацию об отправке
                    print(f"WebSocket: отправка '({current_sector})' в {len(connected_clients)} соединений")
                
                # Очищаем строку и выводим статус для каждого порта
                print("\033[4A", end='')  # Поднимаемся на 4 строки вверх
                
                for i, port in enumerate(PORTS):
                    if port in predictions:
                        pred = predictions[port]
                        status = "ДРОН ОБНАРУЖЕН!" if OPERATING_MODE == "drone" and pred[0] == "class1" else \
                                "ХЛОПОК ОБНАРУЖЕН!" if OPERATING_MODE == "clap" and pred[0] == "Class 2" else \
                                "Фоновый шум"
                        confidence = f"{pred[1]:.1%}".rjust(8)
                        dBFS = f"{pred[2]:+.1f} dBFS".rjust(12)
                        
                        color = "\033[92m" if (OPERATING_MODE == "drone" and pred[0] == "class1") or \
                                             (OPERATING_MODE == "clap" and pred[0] == "Class 2") else "\033[93m"
                        reset = "\033[0m"
                        
                        buffer_size = len(self.audio_buffers[port])
                        
                        print(f"{port:<8} {i+1:<12} {color}{status:<20}{reset} {confidence:<10} {dBFS:<12} {buffer_size:<10}")
                    else:
                        print(f"{port:<8} {i+1:<12} {'Нет данных':<20} {'':<10} {'':<12} {0:<10}")
                
                # Выводим информацию о текущем секторе
                print(f"Текущий сектор: {current_sector}")
                
                # Пауза между обработками
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\nОшибка в процессе обработки аудио: {str(e)}")
                time.sleep(1)  # Пауза при ошибке
                
        print("\nПрерывание процесса обработки аудио.")

    def determine_sector(self):
        """Определение сектора обнаружения на основе уровней сигнала с разных микрофонов"""
        required_devices = PORTS
        
        # Получаем последние предсказания
        predictions = self.last_predictions
        
        # Проверяем наличие всех необходимых устройств
        if not all(dev_id in predictions for dev_id in required_devices):
            return "Не хватает данных с микрофонов"
        
        # Проверка наличия ошибок в данных
        device_classes = {}
        
        activated_devices = []  # Список портов, на которых обнаружен целевой звук
        
        # ОТЛАДКА: Выводим сравнение уровней сигнала по всем микрофонам
        print("\n--- ОТЛАДКА УРОВНЕЙ СИГНАЛА ---")
        for port in PORTS:
            if port in predictions:
                dBFS = predictions[port][2]
                print(f"Микрофон {port}: {dBFS:.1f} dBFS")
        
        for dev_id in required_devices:
            pred = predictions[dev_id]
            if pred[0] == "error":
                return "Ошибка в данных устройства"
            
            # Для режима дрона
            if OPERATING_MODE == "drone":
                try:
                    # Класс "class1" - дрон
                    is_target = (pred[0] == "class1")
                    device_classes[dev_id] = 1 if is_target else 2
                    if is_target:
                        activated_devices.append(dev_id)
                except:
                    return f"Ошибка класса: {pred[0]}"
            # Для режима хлопков
            else:
                try:
                    # Проверяем класс хлопка и уровень громкости
                    # Class 2 - хлопок, проверяем что уверенность выше 30%
                    is_target = (pred[0] == "Class 2" and pred[1] > 0.3)
                    # Также учитываем громкость звука, теперь с порогом -21 dBFS
                    sound_is_loud = (pred[2] > -21)  # Если громкость выше -21 dBFS
                    
                    # Для определения сектора нужен либо высокий уровень уверенности, либо громкий звук
                    is_activated = (is_target or sound_is_loud)
                    device_classes[dev_id] = 1 if is_activated else 2
                    
                    # Добавляем в список активированных устройств
                    if is_activated:
                        activated_devices.append(dev_id)
                        print(f"Порт {dev_id}: Обнаружен целевой звук - уверенность: {pred[1]:.1%}, громкость: {pred[2]:.1f} dBFS")
                    
                except:
                    return f"Ошибка класса: {pred[0]}"

        # Условия для разных секторов
        # Проверяем, есть ли вообще целевой звук
        if all(cls == 2 for cls in device_classes.values()) or not activated_devices:
            # Проверяем, не прошло ли 1 секунда с момента последнего обнаружения
            current_time = time.time()
            if current_time - self.last_detection_time > 1.0 and self.current_active_sector != "Не определен":
                print(f"Сектор сброшен - прошло {current_time - self.last_detection_time:.1f} сек с момента последнего обнаружения")
                self.current_active_sector = "Не определен"
                
                # Отправляем сообщение о сбросе сектора на фронтенд
                if hasattr(self, 'event_loop'):
                    try:
                        # Создаем и отправляем сообщение о сбросе
                        reset_data = {"sector": "Не определен", "timestamp": current_time}
                        future = asyncio.run_coroutine_threadsafe(
                            send_to_clients(reset_data),
                            self.event_loop
                        )
                        # Дожидаемся выполнения с таймаутом 0.5 секунды
                        future.result(timeout=0.5)
                        print("WebSocket: отправлено сообщение о сбросе сектора")
                    except Exception as e:
                        print(f"Ошибка при отправке сообщения о сбросе сектора: {str(e)}")
                        
            return self.current_active_sector  # Возвращаем текущий активный сектор
            
        # Создаем словарь уровней звука только для активированных устройств
        active_sound_levels = {dev_id: predictions[dev_id][2] for dev_id in activated_devices}
        print(f"Активные порты и их уровни: {active_sound_levels}")
            
        # Находим микрофон с максимальным уровнем звука среди активированных
        if active_sound_levels:
            max_device = max(active_sound_levels, key=lambda k: active_sound_levels[k])
            print(f"Выбран порт {max_device} с уровнем {active_sound_levels[max_device]:.1f} dBFS")
            
            # ОТЛАДКА: Проверяем разницу между уровнями микрофонов
            max_level = active_sound_levels[max_device]
            print("\n--- ОТЛАДКА РАЗНИЦЫ УРОВНЕЙ ---")
            for port, level in active_sound_levels.items():
                diff = max_level - level
                print(f"Порт {port}: разница с максимальным {diff:.1f} дБ")
            
            # Если разница между уровнями микрофонов меньше 2 дБ, считаем их равными
            # и выбираем сектор на основе количества активированных микрофонов
            similar_devices = [port for port, level in active_sound_levels.items() 
                               if (max_level - level) < 2.0]
            
            if len(similar_devices) > 1:
                print(f"Обнаружено несколько микрофонов с похожими уровнями: {similar_devices}")
                
                # Если активны и верхние, и нижние микрофоны с похожими уровнями,
                # выбираем сектор по расположению
                upper_mics = [port for port in similar_devices if port in [5000, 5001]]
                lower_mics = [port for port in similar_devices if port in [5002, 5003]]
                
                if upper_mics and lower_mics:
                    print("Активны и верхние, и нижние микрофоны - выбираем на основе среднего уровня")
                    
                    avg_upper = sum(active_sound_levels[port] for port in upper_mics) / len(upper_mics)
                    avg_lower = sum(active_sound_levels[port] for port in lower_mics) / len(lower_mics)
                    
                    print(f"Средний уровень верхних: {avg_upper:.1f} дБ, нижних: {avg_lower:.1f} дБ")
                    
                    if avg_upper > avg_lower:
                        max_device = upper_mics[0]  # Берем первый из верхних
                        print(f"Выбираем верхние микрофоны, новый порт: {max_device}")
                    else:
                        max_device = lower_mics[0]  # Берем первый из нижних
                        print(f"Выбираем нижние микрофоны, новый порт: {max_device}")
        else:
            # Если нет активированных устройств, сохраняем текущий сектор
            return self.current_active_sector
        
        # Определяем новый сектор - используем такие же названия, как и раньше для совместимости с фронтендом
        if max_device == 5000:  # Микрофон 1
            new_sector = "СВЕРХУ-СЛЕВА"  # Именно так ожидает фронтенд - с дефисом!
        elif max_device == 5001:  # Микрофон 2
            new_sector = "СВЕРХУ-СПРАВА"  # Именно так ожидает фронтенд - с дефисом!
        elif max_device == 5002 or max_device == 5003:  # Микрофоны 3 и 4
            new_sector = "СНИЗУ"  # Именно так ожидает фронтенд - без направления!
        else:
            new_sector = "Не определен"
            
        print(f"Определен новый сектор: {new_sector}")
            
        # Обновляем время последнего обнаружения и сектор
        self.last_detection_time = time.time()
        self.current_active_sector = new_sector
            
        return new_sector

# Обработка аргументов командной строки
def parse_arguments():
    parser = argparse.ArgumentParser(description='Система обнаружения звуков с нейронной сетью')
    parser.add_argument('--mode', type=str, choices=['drone', 'clap'], default='drone',
                        help='Режим работы: drone - обнаружение дронов, clap - демо-режим с хлопками')
    return parser.parse_args()

# Главная функция
if __name__ == "__main__":
    # Парсинг аргументов
    args = parse_arguments()
    
    # Выбор режима работы
    if args.mode == 'clap':
        print("\n" + "="*60)
        print("ЗАПУСК В ДЕМО-РЕЖИМЕ С РАСПОЗНАВАНИЕМ ХЛОПКОВ")
        print("="*60 + "\n")
        
        # Проверка наличия файлов модели
        tflite_path = find_model_file(TFLITE_MODEL_PATH)
        labels_path = find_model_file(LABELS_PATH)
        
        if not tflite_path or not labels_path:
            print("\nВНИМАНИЕ: Не все необходимые файлы найдены для демо-режима.")
            print("Пожалуйста, запустите скрипт copy_model_files.bat для автоматического копирования файлов")
            print("или вручную скопируйте файлы soundclassifier_with_metadata.tflite и labels.txt")
            print("в директорию с программой.")
            print("\nПереключение на стандартный режим обнаружения дронов...")
            args.mode = 'drone'
    else:
        print("\n" + "="*60)
        print("ЗАПУСК В РЕЖИМЕ ОБНАРУЖЕНИЯ ДРОНОВ")
        print("="*60 + "\n")
    
    # Создание и запуск классификатора
    classifier = SoundClassifier(mode=args.mode)
    
    try:
        # Просто ждем, пока потоки работают
        while classifier.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем")
    finally:
        # Корректное завершение
        classifier.running = False
        print("Закрытие сокетов...")
        for sock in classifier.sockets.values():
            try:
                sock.close()
            except:
                pass
        print("Выход из программы")