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

# Конфигурация
PORTS = [5000, 5001, 5002, 5003]
TARGET_SAMPLE_RATE = 44100
N_MFCC = 40
MODEL_PATH = 'sound_classifier_model.pth'
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

# Функция для отправки данных через WebSocket
async def send_to_clients(data):
    if connected_clients:  # Проверка наличия подключенных клиентов
        # Создаем JSON-объект для отправки
        message = json.dumps(data)
        # Отправляем всем подключенным клиентам
        await asyncio.gather(
            *[client.send(message) for client in connected_clients],
            return_exceptions=True
        )
        print(f"WebSocket: отправлены данные о секторе '{data['sector']}' {len(connected_clients)} клиентам")
    else:
        print("WebSocket: нет подключенных клиентов")

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
    def __init__(self):
        self.le = LabelEncoder()
        self.model = None
        self.running = True
        self.lock = threading.Lock()
        self.audio_buffers = {port: np.array([], dtype=np.float32) for port in PORTS}
        self.last_predictions = {}
        self.sample_rates = {}
        self.sockets = {}  # Для хранения сокетов
        
        # Инициализация путей к данным
        self.data_dir = DATA_DIRS['train']
        self.valid_dir = DATA_DIRS['valid']
        self.test_dir = DATA_DIRS['test']
        
        if not self.load_model():  # Попытка загрузки модели
            self.init_training()    # Запуск обучения если модель не найдена
            
        self.init_network()

    # 🔄 Перемещен метод save_model выше load_model
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
        """Инициализация сетевых сокетов и запуск WebSocket сервера"""
        self.sockets = {}
        
        for port in PORTS:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('0.0.0.0', port))
                sock.setblocking(0)
                self.sockets[port] = sock
                print(f"Сокет успешно инициализирован на порту {port}")
            except Exception as e:
                print(f"Ошибка инициализации сокета на порту {port}: {str(e)}")
                if port in self.sockets:
                    del self.sockets[port]
        
        if not self.sockets:
            raise Exception("Не удалось инициализировать ни один сокет")
            
        # Запуск WebSocket сервера в отдельном потоке
        self._run_websocket_server()
    
    def _run_websocket_server(self):
        """Запуск WebSocket сервера в отдельном потоке"""
        # Создаем новый event loop для WebSocket сервера
        self.event_loop = asyncio.new_event_loop()
        # Запускаем WebSocket сервер в отдельном потоке
        websocket_thread = threading.Thread(
            target=self._start_websocket_loop,
            daemon=True
        )
        websocket_thread.start()
    
    def _start_websocket_loop(self):
        """Функция для запуска event loop в отдельном потоке"""
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_until_complete(start_websocket_server())

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
        try:
            if len(audio) == 0:
                return {'class': 'error', 'confidence': 0, 'dBFS': -np.inf}
            
            # Проверка на тишину
            if np.all(np.abs(audio) < 1e-6):
                return {'class': 'silence', 'confidence': 0, 'dBFS': -np.inf}

            # Нормализация аудио
            audio = librosa.util.normalize(audio)
            
            # Вычисление RMS энергии
            rms = np.sqrt(np.mean(audio**2))
            dBFS = 20 * np.log10(rms) if rms > 0 else -np.inf
            
            # Если уровень звука слишком низкий, считаем это тишиной
            if dBFS < -50:  # Уменьшили порог тишины
                return {'class': 'silence', 'confidence': 0, 'dBFS': dBFS}
            
            # Вычисление MFCC
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=TARGET_SAMPLE_RATE,
                n_mfcc=40,
                n_fft=2048,
                hop_length=512
            )
            
            # Усредняем MFCC по времени
            features = np.mean(mfcc.T, axis=0)
            
            # Проверка на валидность признаков
            if np.isnan(features).any() or np.isinf(features).any():
                return {'class': 'error', 'confidence': 0, 'dBFS': dBFS}
            
            # Предсказание
            self.model.eval()
            with torch.no_grad():
                inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                proba = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(proba, 1)
            
            # Дополнительные проверки для уменьшения ложных срабатываний
            # 1. Проверка спектральных характеристик
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=TARGET_SAMPLE_RATE))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=TARGET_SAMPLE_RATE))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=TARGET_SAMPLE_RATE))
            
            # 2. Проверка временных характеристик
            zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y=audio))
            
            # 3. Проверка ритмических характеристик
            onset_env = librosa.onset.onset_strength(y=audio, sr=TARGET_SAMPLE_RATE)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=TARGET_SAMPLE_RATE)[0]
            
            # Корректировка уверенности на основе характеристик
            # 1. Уровень звука
            if dBFS < -30:  # Дрон обычно громче
                conf = conf * 0.3
            elif dBFS > -10:  # Слишком громкий звук
                conf = conf * 0.7
            
            # 2. Спектральные характеристики
            if spectral_centroid < 2000 or spectral_centroid > 8000:  # Дрон обычно в этом диапазоне
                conf = conf * 0.5
            if spectral_bandwidth < 1000 or spectral_bandwidth > 5000:  # Дрон обычно в этом диапазоне
                conf = conf * 0.5
            if spectral_rolloff < 3000 or spectral_rolloff > 10000:  # Дрон обычно в этом диапазоне
                conf = conf * 0.5
            
            # 3. Временные характеристики
            if zero_crossings < 0.05 or zero_crossings > 0.2:  # Дрон обычно в этом диапазоне
                conf = conf * 0.5
            
            # 4. Ритмические характеристики
            if tempo < 100 or tempo > 300:  # Дрон обычно в этом диапазоне
                conf = conf * 0.5
            
            # 5. Проверка стабильности сигнала
            signal_variance = np.var(audio)
            if signal_variance < 0.01 or signal_variance > 0.5:  # Дрон обычно в этом диапазоне
                conf = conf * 0.5
            
            # Ограничиваем уверенность в пределах [0, 1]
            conf = min(max(conf.item(), 0), 1)
            
            # Повышаем порог уверенности для определения дрона
            class_label = 'class1' if pred.item() == 1 and conf > 0.85 else 'class2'  # Повышенный порог
            
            return {
                'class': class_label,
                'confidence': conf,
                'dBFS': dBFS
            }
            
        except Exception as e:
            print(f"Ошибка предсказания: {str(e)}")
            return {'class': 'error', 'confidence': 0, 'dBFS': -np.inf}

    def process_audio(self):
        while self.running:
            try:
                predictions = {}
                for port in PORTS:
                    with self.lock:
                        buffer = self.audio_buffers[port]
                    # Накопление 2 секунд аудио перед обработкой
                        if len(buffer) >= BUFFER_SIZE:
                            audio = buffer[-BUFFER_SIZE:]  # Берем последние 2 секунды
                            self.audio_buffers[port] = np.array([], dtype=np.float32)
                            predictions[port] = self.predict(audio)
                
                if predictions:
                    self.last_predictions = {
                        i+1: pred for i, (port, pred) in enumerate(predictions.items())
                    }
                    
                    sector = self.determine_sector()
                    print("\n" + "="*60)
                    print(f"{'Система мониторинга дронов':^60}")
                    print("="*60)
                    print(f"\nОпределенный сектор: \033[1m{sector}\033[0m\n")
                    
                    # Отправляем данные через WebSocket
                    data_to_send = {
                        "sector": sector,
                        "timestamp": time.time()
                    }
                    # Используем asyncio для отправки данных
                    if hasattr(self, 'event_loop'):
                        asyncio.run_coroutine_threadsafe(
                            send_to_clients(data_to_send),
                            self.event_loop
                        )
                    else:
                        print("Ошибка: event_loop не доступен для отправки данных")
                    
                    for port, pred in predictions.items():
                        status = "ДРОН ОБНАРУЖЕН!" if pred['class'] == 'class1' else "Фоновый шум"
                        confidence = f"{pred['confidence']:.1%}".rjust(8)
                        dBFS = f"{pred['dBFS']:+.1f} dBFS".rjust(12)
                        
                        color = "\033[92m" if pred['class'] == 'class1' else "\033[93m"
                        reset = "\033[0m"
                        
                        print(f"Порт {port}:")
                        print(f"{color}├─ Статус: {status}{reset}")
                        print(f"├─ Уровень достоверности: {confidence}")
                        print(f"└─ Уровень звука:    {dBFS}")
                    print("="*60)
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                print(f"Ошибка обработки: {str(e)}")

    def determine_sector(self):
        """Определение сектора по предсказаниям"""
        predictions = self.last_predictions
        required_devices = [1, 2, 3, 4]

        for dev_id in required_devices:
            if dev_id not in predictions:
                return "Не определено (недостаточно данных)"

        device_classes = {}
        for dev_id in required_devices:
            pred = predictions[dev_id]
            if pred['class'] == 'error':
                return "Ошибка в данных устройства"
            
            try:
                class_num = int(pred['class'].replace('class', ''))
                device_classes[dev_id] = class_num
            except:
                return f"Ошибка класса: {pred['class']}"

        conditions = [
            (device_classes[1] == 1 and device_classes[2] == 1 and 
             device_classes[4] == 1 and device_classes[3] == 2, "СВЕРХУ-СЛЕВА"),
            (device_classes[1] == 1 and device_classes[2] == 1 and 
             device_classes[3] == 1 and device_classes[4] == 2, "СВЕРХУ-СПРАВА"),
            (device_classes[1] == 1 and device_classes[3] == 1 and 
             device_classes[4] == 1 and device_classes[2] == 2, "СНИЗУ"),
            (device_classes[2] == 1 and device_classes[3] == 1 and 
             device_classes[4] == 1 and device_classes[1] == 2, "Ошибка конфигурации"),
            (device_classes[2] == 2 and device_classes[3] == 2 and 
             device_classes[4] == 2 and device_classes[1] == 1, "Ошибка конфигурации")
        ]

        for condition, sector in conditions:
            if condition:
                return sector

        if all(cls == 1 for cls in device_classes.values()):
            sound_levels = {dev_id: predictions[dev_id]['dBFS'] for dev_id in required_devices}
            max_device = max(sound_levels, key=lambda k: sound_levels[k])
            remaining_devices = sorted([d for d in required_devices if d != max_device])
            
            combination = ''.join(map(str, remaining_devices))
            sectors = {
                '123': "СВЕРХУ-СПРАВА",
                '134': "СНИЗУ",
                '124': "СВЕРХУ-СЛЕВА",
                '234': "ОШИБКА КОНФИГУРАЦИИ"
            }
            
            return sectors.get(combination, f"Неизвестная комбинация: {combination}")

        return "Неопределенный сектор"

if __name__ == "__main__":
    classifier = SoundClassifier()
    
    # Запуск сетевых потоков
    listeners = []
    for port in PORTS:
        thread = threading.Thread(
            target=classifier.network_listener,
            args=(port,),
            daemon=True
        )
        thread.start()
        listeners.append(thread)
    
    # Основной цикл обработки
    try:
        classifier.process_audio()
    finally:
        print("\nОстановка сервера...")
        for sock in classifier.sockets.values():
            sock.close()
        print("Сервер успешно остановлен.")