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
import cv2
import torch.nn.functional as F
import random
from localization import DroneLocalizer, calculate_tdoa  # Исправленный импорт локализатора

# Функции аугментации данных
def add_noise(audio, noise_level=0.005):
    """Добавление случайного шума"""
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

def change_volume(audio, volume_range=(0.5, 1.5)):
    """Изменение громкости"""
    volume = random.uniform(*volume_range)
    return audio * volume

def time_shift(audio, sr, max_shift=0.1):
    """Сдвиг по времени"""
    shift = int(random.uniform(-max_shift, max_shift) * sr)
    if shift > 0:
        return np.pad(audio[shift:], (0, shift), mode='constant')
    else:
        return np.pad(audio[:shift], (-shift, 0), mode='constant')

def pitch_shift(audio, sr, n_steps=(-2, 2)):
    """Изменение частоты"""
    steps = random.uniform(*n_steps)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)

def augment_audio(audio, sr):
    """Применение случайной аугментации"""
    aug_type = random.choice(['noise', 'volume', 'time_shift', 'pitch_shift'])
    
    if aug_type == 'noise':
        return add_noise(audio)
    elif aug_type == 'volume':
        return change_volume(audio)
    elif aug_type == 'time_shift':
        return time_shift(audio, sr)
    elif aug_type == 'pitch_shift':
        return pitch_shift(audio, sr)
    
    return audio

# Определение класса CNN
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        # Сверточные слои
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Пулинг
        self.pool = nn.MaxPool2d(2, 2)
        
        # Вычисляем размер после сверточных слоев
        # 128 -> 64 -> 32 -> 16
        self.fc_input_size = 128 * 16 * 16
        
        # Полносвязные слои
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Проверяем размерность входных данных
        batch_size = x.size(0)
        
        # Сверточные слои с активацией и пулингом
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Выравнивание для полносвязных слоев
        x = x.view(batch_size, -1)  # Используем batch_size вместо -1
        
        # Проверяем размерность перед полносвязным слоем
        if x.size(1) != self.fc_input_size:
            raise RuntimeError(f"Размерность после сверточных слоев ({x.size(1)}) не соответствует ожидаемой ({self.fc_input_size})")
        
        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

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

# Параметры для спектрограмм
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SPECTROGRAM_SIZE = (128, 128)  # Размер спектрограммы (высота, ширина)

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
        self.sockets = {}
        
        # Инициализация локализатора
        self.localizer = DroneLocalizer()
        
        # Добавляем таймер для отслеживания времени последнего обнаружения
        self.last_detection_time = 0
        self.current_active_sector = "Не определен"
        
        # Инициализация путей к данным
        self.data_dir = DATA_DIRS['train']
        self.valid_dir = DATA_DIRS['valid']
        self.test_dir = DATA_DIRS['test']
        
        print(f"Инициализация в режиме: {OPERATING_MODE}")
        
        if OPERATING_MODE == "drone":
            if not self.load_model():
                self.init_training()
        else:
            if not TENSORFLOW_AVAILABLE:
                print("Критическая ошибка: TensorFlow не установлен, но выбран режим хлопка!")
                print("Пожалуйста, установите TensorFlow: pip install tensorflow")
                print("Или запустите install_dependencies.bat")
                print("Переключение на стандартный режим дрона...")
                OPERATING_MODE = "drone"
                if not self.load_model():
                    self.init_training()
            elif not self.load_tflite_model():
                print("Критическая ошибка: не удалось загрузить TensorFlow модель для режима хлопка!")
                print("Переключение на стандартный режим дрона...")
                OPERATING_MODE = "drone"
                if not self.load_model():
                    self.init_training()
            
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

    def save_model(self):
        """Сохранение модели на диск"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'le_classes': self.le.classes_,
                'sample_rates': self.sample_rates,
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
                num_classes = checkpoint['num_classes']
                self.model = CNN(num_classes)
                
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
            self.train(num_epochs=10)  # Увеличиваем количество эпох до 10
            self.save_model()
            print("\nОбучение успешно завершено!\n")
            self.load_model()
        except Exception as e:
            print(f"Критическая ошибка при обучении: {str(e)}")
            exit(1)

    def load_data(self):
        def load_from_dir(dir_path, augment=False):
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
                            
                            # Загрузка аудио
                            audio, sr = librosa.load(file_path, sr=None, mono=True)
                            if sr != TARGET_SAMPLE_RATE:
                                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
                                
                            # Создание спектрограммы для оригинального аудио
                            mel_spec = librosa.feature.melspectrogram(
                                    y=audio,
                                    sr=TARGET_SAMPLE_RATE,
                                n_fft=N_FFT,
                                hop_length=HOP_LENGTH,
                                n_mels=N_MELS
                            )
                            
                            # Преобразование в децибелы
                            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                            
                            # Замена NaN и inf на 0
                            mel_spec_db = np.nan_to_num(mel_spec_db, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            # Нормализация в диапазон [0, 1]
                            min_val = np.min(mel_spec_db)
                            max_val = np.max(mel_spec_db)
                            if max_val > min_val:  # Избегаем деления на ноль
                                mel_spec_db = (mel_spec_db - min_val) / (max_val - min_val)
                            else:
                                mel_spec_db = np.zeros_like(mel_spec_db)
                            
                            # Изменение размера до SPECTROGRAM_SIZE
                            mel_spec_db = cv2.resize(mel_spec_db, SPECTROGRAM_SIZE)
                            
                            # Проверка на NaN после всех преобразований
                            if np.isnan(mel_spec_db).any():
                                print(f"Предупреждение: NaN обнаружены в {file_path} после обработки")
                                continue
                            
                            # Добавление размерности канала
                            mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
                            
                            X.append(mel_spec_db)
                            y.append(label)
                            
                            # Аугментация данных для тренировочного набора
                            if augment:
                                # Создаем 3 аугментированных версии каждого файла
                                for _ in range(3):
                                    # Применяем случайную аугментацию
                                    augmented_audio = augment_audio(audio, TARGET_SAMPLE_RATE)
                                    
                                    # Создаем спектрограмму для аугментированного аудио
                                    aug_mel_spec = librosa.feature.melspectrogram(
                                        y=augmented_audio,
                                        sr=TARGET_SAMPLE_RATE,
                                        n_fft=N_FFT,
                                        hop_length=HOP_LENGTH,
                                        n_mels=N_MELS
                                    )
                                    
                                    # Преобразование в децибелы
                                    aug_mel_spec_db = librosa.power_to_db(aug_mel_spec, ref=np.max)
                                    
                                    # Замена NaN и inf на 0
                                    aug_mel_spec_db = np.nan_to_num(aug_mel_spec_db, nan=0.0, posinf=0.0, neginf=0.0)
                                    
                                    # Нормализация
                                    min_val = np.min(aug_mel_spec_db)
                                    max_val = np.max(aug_mel_spec_db)
                                    if max_val > min_val:
                                        aug_mel_spec_db = (aug_mel_spec_db - min_val) / (max_val - min_val)
                                    else:
                                        aug_mel_spec_db = np.zeros_like(aug_mel_spec_db)
                                    
                                    # Изменение размера
                                    aug_mel_spec_db = cv2.resize(aug_mel_spec_db, SPECTROGRAM_SIZE)
                                    
                                    # Проверка на NaN
                                    if np.isnan(aug_mel_spec_db).any():
                                        continue
                                    
                                    # Добавление размерности канала
                                    aug_mel_spec_db = np.expand_dims(aug_mel_spec_db, axis=0)
                                    
                                    X.append(aug_mel_spec_db)
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
            self.X, self.y = load_from_dir(self.data_dir, augment=True)  # Аугментация только для тренировочных данных
            print(f"Метки классов: {np.unique(self.y)}")
            self.y_encoded = self.le.fit_transform(self.y)
            print(f"Закодированные метки: {np.unique(self.y_encoded)}")
            
            print("\nЗагрузка валидационных данных...")
            self.X_valid, self.y_valid = load_from_dir(self.valid_dir, augment=False)
            print(f"Валидационные метки: {np.unique(self.y_valid)}")
            self.y_encoded_valid = self.le.transform(self.y_valid)
            print(f"Закодированные валидационные метки: {np.unique(self.y_encoded_valid)}")
            
            print("\nЗагрузка тестовых данных...")
            self.X_test, self.y_test = load_from_dir(self.test_dir, augment=False)
            self.y_encoded_test = self.le.transform(self.y_test)
            
            # Проверка на NaN в загруженных данных
            if np.isnan(self.X).any():
                print("Предупреждение: NaN обнаружены в тренировочных данных")
                self.X = np.nan_to_num(self.X, nan=0.0)
            
            if np.isnan(self.X_valid).any():
                print("Предупреждение: NaN обнаружены в валидационных данных")
                self.X_valid = np.nan_to_num(self.X_valid, nan=0.0)
            
            if np.isnan(self.X_test).any():
                print("Предупреждение: NaN обнаружены в тестовых данных")
                self.X_test = np.nan_to_num(self.X_test, nan=0.0)
            
            print("\nСтатистика датасета:")
            self._print_dataset_stats("Тренировочные", self.y)
            self._print_dataset_stats("Валидационные", self.y_valid)
            self._print_dataset_stats("Тестовые", self.y_test)
            
            # Проверяем размерности данных
            print(f"\nРазмерность тренировочных данных: {self.X.shape}")
            print(f"Размерность валидационных данных: {self.X_valid.shape}")
            print(f"Размерность тестовых данных: {self.X_test.shape}")
            
            # Проверяем метки классов
            print(f"\nУникальные метки в тренировочных данных: {np.unique(self.y_encoded)}")
            print(f"Уникальные метки в валидационных данных: {np.unique(self.y_encoded_valid)}")
            print(f"Количество классов: {len(self.le.classes_)}")
            
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
        """Создание сверточной нейронной сети"""
        # Создаем модель
        num_classes = len(self.le.classes_)
        print(f"\nСоздание модели с {num_classes} классами")
        print(f"Метки классов: {self.le.classes_}")
        
        self.model = CNN(num_classes)
        
        # Выводим информацию о модели
        print("\nАрхитектура модели:")
        print(self.model)
        
        # Инициализация весов
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, num_epochs=5):
        # Устанавливаем размер батча
        batch_size = 8  # Уменьшаем размер батча
        
        # Проверяем размерности данных
        print(f"\nРазмерность тренировочных данных: {self.X.shape}")
        print(f"Размерность валидационных данных: {self.X_valid.shape}")
        
        # Проверяем метки классов
        print(f"\nУникальные метки в тренировочных данных: {np.unique(self.y_encoded)}")
        print(f"Уникальные метки в валидационных данных: {np.unique(self.y_encoded_valid)}")
        print(f"Количество классов: {len(self.le.classes_)}")
        
        # Создаем DataLoader для тренировочных данных
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(self.X, dtype=torch.float32),
            torch.tensor(self.y_encoded, dtype=torch.long)
        )
        print(f"\nРазмер тренировочного датасета: {len(train_dataset)}")
        print(f"Размерность X в датасете: {train_dataset.tensors[0].shape}")
        print(f"Размерность y в датасете: {train_dataset.tensors[1].shape}")
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Создаем DataLoader для валидационных данных
        valid_dataset = torch.utils.data.TensorDataset(
            torch.tensor(self.X_valid, dtype=torch.float32),
            torch.tensor(self.y_encoded_valid, dtype=torch.long)
        )
        print(f"\nРазмер валидационного датасета: {len(valid_dataset)}")
        print(f"Размерность X в валидационном датасете: {valid_dataset.tensors[0].shape}")
        print(f"Размерность y в валидационном датасете: {valid_dataset.tensors[1].shape}")
        
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Проверяем первый батч
        print("\nПроверка первого батча:")
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"\nБатч {batch_idx + 1}:")
            print(f"Размерность данных: {data.shape}")
            print(f"Размерность меток: {target.shape}")
            print(f"Уникальные метки в батче: {torch.unique(target)}")
            
            # Проверяем размерность после каждого слоя
            with torch.no_grad():
                x = self.model.conv1(data)
                x = self.model.pool(F.relu(self.model.bn1(x)))
                print(f"После conv1 + pool: {x.shape}")
                
                x = self.model.conv2(x)
                x = self.model.pool(F.relu(self.model.bn2(x)))
                print(f"После conv2 + pool: {x.shape}")
                
                x = self.model.conv3(x)
                x = self.model.pool(F.relu(self.model.bn3(x)))
                print(f"После conv3 + pool: {x.shape}")
                
                x = x.view(batch_size, -1)
                print(f"После view: {x.shape}")
                
                x = self.model.fc1(x)
                print(f"После fc1: {x.shape}")
                
                x = self.model.fc2(x)
                print(f"После fc2: {x.shape}")
                
                # Проверяем размерность выходных данных
                if x.shape[1] != len(self.le.classes_):
                    raise RuntimeError(f"Неверное количество выходных классов. Ожидается {len(self.le.classes_)}, получено {x.shape[1]}")
            
            # Проверяем только первый батч
            break
        
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        learning_rates = []
        
        # Очищаем кэш CUDA если доступен
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            for epoch in range(num_epochs):
                if not self.running:
                    raise KeyboardInterrupt
                
                # Обучение
                self.model.train()
                epoch_train_loss = 0
                epoch_train_correct = 0
                epoch_train_total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    
                    # Проверяем размерности батча
                    if data.shape[1] != 1:  # Проверяем количество каналов
                        print(f"Ошибка: Неверное количество каналов в батче. Ожидается 1, получено {data.shape[1]}")
                        continue
                        
                    if data.shape[2] != SPECTROGRAM_SIZE[0] or data.shape[3] != SPECTROGRAM_SIZE[1]:
                        print(f"Ошибка: Неверный размер спектрограммы. Ожидается {SPECTROGRAM_SIZE}, получено {data.shape[2:]}")
                        continue
                    
                    # Очищаем кэш каждые 5 батчей
                    if batch_idx % 5 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Проверяем на NaN перед forward pass
                    if torch.isnan(data).any():
                        print(f"Обнаружены NaN в данных на батче {batch_idx}")
                        continue
                    
                    outputs = self.model(data)
                    
                    # Проверяем размерность выходных данных
                    if outputs.shape[1] != len(self.le.classes_):
                        print(f"Ошибка: Неверное количество выходных классов. Ожидается {len(self.le.classes_)}, получено {outputs.shape[1]}")
                        continue
                    
                    # Проверяем на NaN после forward pass
                    if torch.isnan(outputs).any():
                        print(f"Обнаружены NaN в выходных данных на батче {batch_idx}")
                        continue
                    
                    loss = self.criterion(outputs, target)
                    
                    loss.backward()
                    
                    # Обрезаем градиенты для предотвращения взрыва градиентов
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    epoch_train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_train_total += target.size(0)
                    epoch_train_correct += (predicted == target).sum().item()
                    
                    # Очищаем память
                    del outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Расчет средней потери и точности для эпохи
                if epoch_train_total > 0:
                    avg_train_loss = epoch_train_loss / len(train_loader)
                    train_accuracy = 100 * epoch_train_correct / epoch_train_total
                else:
                    avg_train_loss = float('inf')
                    train_accuracy = 0.0
                
                train_losses.append(avg_train_loss)
                train_accuracies.append(train_accuracy)
            
                # Валидация
                self.model.eval()
                epoch_valid_loss = 0
                epoch_valid_correct = 0
                epoch_valid_total = 0
                
                with torch.no_grad():
                    for data, target in valid_loader:
                        # Проверяем размерности валидационного батча
                        if data.shape[1] != 1:
                            print(f"Ошибка: Неверное количество каналов в валидационном батче. Ожидается 1, получено {data.shape[1]}")
                            continue
                            
                        if data.shape[2] != SPECTROGRAM_SIZE[0] or data.shape[3] != SPECTROGRAM_SIZE[1]:
                            print(f"Ошибка: Неверный размер спектрограммы в валидационном батче. Ожидается {SPECTROGRAM_SIZE}, получено {data.shape[2:]}")
                            continue
                        
                        # Проверяем на NaN в валидационных данных
                        if torch.isnan(data).any():
                            print("Обнаружены NaN в валидационных данных")
                            continue
                        
                        outputs = self.model(data)
                        
                        # Проверяем размерность выходных данных
                        if outputs.shape[1] != len(self.le.classes_):
                            print(f"Ошибка: Неверное количество выходных классов в валидации. Ожидается {len(self.le.classes_)}, получено {outputs.shape[1]}")
                            continue
                        
                        loss = self.criterion(outputs, target)
                        
                        epoch_valid_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        epoch_valid_total += target.size(0)
                        epoch_valid_correct += (predicted == target).sum().item()
                        
                        # Очищаем память
                        del outputs, loss
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Расчет средней потери и точности для валидации
                if epoch_valid_total > 0:
                    avg_valid_loss = epoch_valid_loss / len(valid_loader)
                    valid_accuracy = 100 * epoch_valid_correct / epoch_valid_total
                else:
                    avg_valid_loss = float('inf')
                    valid_accuracy = 0.0
                
                valid_losses.append(avg_valid_loss)
                valid_accuracies.append(valid_accuracy)
            
                learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
                print(f"Эпоха [{epoch+1}/{num_epochs}] | "
                    f"Потеря: {avg_train_loss:.4f} | Валидация: {avg_valid_loss:.4f} | "
                    f"Точность: {train_accuracy:.2f}% | Валидационная точность: {valid_accuracy:.2f}%")
                
                # Очищаем память после каждой эпохи
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("\nОбучение прервано пользователем")
            self.running = False
        
        finally:
            # Проверяем, что у нас есть данные для анализа
            if not train_accuracies or not valid_accuracies:
                print("\n⚠️ Обучение было прервано до завершения первой эпохи")
                print("Нет данных для анализа")
                return
                
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
    
        # Проверяем, что списки не пустые
        if not train_accuracies or not valid_accuracies:
            print("⚠️ Ошибка: Нет данных для анализа (пустые списки точности)")
            return
            
        # Проверяем, что все списки имеют одинаковую длину
        if not all(len(lst) == len(train_accuracies) for lst in [valid_accuracies, train_losses, valid_losses, learning_rates]):
            print("⚠️ Ошибка: Несоответствие длин списков данных")
            return
            
        # Анализ точности
        try:
            max_train_acc = max(train_accuracies)
            min_train_acc = min(train_accuracies)
            final_train_acc = train_accuracies[-1]
        except ValueError:
            print("⚠️ Ошибка: Нет данных о тренировочной точности")
            return
            
        try:
            max_valid_acc = max(valid_accuracies)
            min_valid_acc = min(valid_accuracies)
            final_valid_acc = valid_accuracies[-1]
        except ValueError:
            print("⚠️ Ошибка: Нет данных о валидационной точности")
            return
    
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
        """Классификация звука дрона с использованием CNN и спектрограмм"""
        # Проверка длины аудио
        if len(audio) < TARGET_SAMPLE_RATE:
            return "Фоновый шум", 0.0  # Недостаточно данных
            
        try:
            # Создание спектрограммы
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=TARGET_SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS
            )
            
            # Преобразование в децибелы
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Замена NaN и inf на 0
            mel_spec_db = np.nan_to_num(mel_spec_db, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Нормализация в диапазон [0, 1]
            min_val = np.min(mel_spec_db)
            max_val = np.max(mel_spec_db)
            if max_val > min_val:  # Избегаем деления на ноль
                mel_spec_db = (mel_spec_db - min_val) / (max_val - min_val)
            else:
                mel_spec_db = np.zeros_like(mel_spec_db)
            
            # Изменение размера до SPECTROGRAM_SIZE
            mel_spec_db = cv2.resize(mel_spec_db, SPECTROGRAM_SIZE)
            
            # Проверка на NaN после всех преобразований
            if np.isnan(mel_spec_db).any():
                print("Предупреждение: NaN обнаружены в спектрограмме")
                return "Фоновый шум", 0.0
            
            # Добавление размерности канала и батча
            mel_spec_db = np.expand_dims(mel_spec_db, axis=0)  # Добавляем канал
            mel_spec_db = np.expand_dims(mel_spec_db, axis=0)  # Добавляем батч
            
            # Преобразование в тензор PyTorch
            features = torch.tensor(mel_spec_db, dtype=torch.float32)
            
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
            print(f"Ошибка предсказания (CNN): {str(e)}")
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
        print("\n=== ЗАПУСК СИСТЕМЫ ОБНАРУЖЕНИЯ ДРОНОВ ===")
        
        # Вывод заголовка таблицы статуса
        print("\n" + "="*100)
        print(f"{'Микрофон':<12} {'Порт':<8} {'Статус':<20} {'Уверенность':<12} {'Уровень звука':<15} {'Размер буфера':<15}")
        print("="*100)
        print("\n" * 5)  # Добавляем пустые строки для начального вывода
            
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
                        if len(buffer) > 0:
                            if np.isnan(buffer).any() or np.isinf(buffer).any():
                                print(f"⚠️ Некорректные значения в буфере порта {port}")
                                buffer = np.nan_to_num(buffer, nan=0.0, posinf=0.0, neginf=0.0)
                            current_buffers[port] = buffer.copy()
                        else:
                            current_buffers[port] = np.array([], dtype=np.float32)
                
                # Обрабатываем каждый буфер отдельно
                predictions = {}
                
                for port, buffer in current_buffers.items():
                    if len(buffer) == 0:
                        predictions[port] = self.last_predictions.get(port, ("Фоновый шум", 0.0, -np.inf))
                        continue
                    
                    class_name, confidence = self.predict(buffer)
                    
                    if len(buffer) > 0:
                        rms = np.sqrt(np.mean(buffer**2))
                        dBFS = 20 * np.log10(rms) if rms > 0 else -np.inf
                    else:
                        dBFS = -np.inf
                    
                    predictions[port] = (class_name, confidence, dBFS)
                    self.last_predictions[port] = predictions[port]
                
                # Определяем текущий активный сектор
                localization_result = self.determine_sector()
                current_sector = localization_result["sector"]
                
                # Отправляем информацию через WebSocket при необходимости
                current_time = time.time()
                should_update = (current_time - last_websocket_update >= 0.1) or (current_sector != self.current_active_sector)
                
                if should_update and hasattr(self, 'event_loop'):
                    last_websocket_update = current_time
                    data_to_send = {
                        "sector": current_sector,
                        "coordinates": localization_result["coordinates"],
                        "error": localization_result["error"],
                        "timestamp": current_time
                    }
                    
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            send_to_clients(data_to_send),
                            self.event_loop
                        )
                        future.result(timeout=0.5)
                    except Exception as e:
                        print(f"⚠️ Ошибка WebSocket: {str(e)}")
                
                # Очищаем предыдущий вывод
                print("\033[5A", end='')  # Поднимаемся на 5 строк вверх
                
                # Выводим статус каждого микрофона
                for i, port in enumerate(PORTS, 1):
                    if port in predictions:
                        pred = predictions[port]
                        status = "🔴 ДРОН!" if OPERATING_MODE == "drone" and pred[0] == "class1" else \
                                "🔔 ХЛОПОК!" if OPERATING_MODE == "clap" and pred[0] == "Class 2" else \
                                "✓ Фоновый шум"
                        confidence = f"{pred[1]*100:.1f}%".rjust(10)
                        dBFS = f"{pred[2]:+.1f} dB".rjust(12)
                        
                        # Цветовое оформление
                        if (OPERATING_MODE == "drone" and pred[0] == "class1") or \
                           (OPERATING_MODE == "clap" and pred[0] == "Class 2"):
                            color = "\033[91m"  # Красный для обнаружения
                        elif pred[2] > -30:  # Если уровень звука выше -30 дБ
                            color = "\033[93m"  # Желтый для заметного звука
                        else:
                            color = "\033[92m"  # Зеленый для фонового шума
                        reset = "\033[0m"
                        
                        buffer_size = len(self.audio_buffers[port])
                        print(f"{f'Микрофон {i}':<12} {port:<8} {color}{status:<20}{reset} {confidence:<12} {dBFS:<15} {buffer_size:<15}")
                    else:
                        print(f"{f'Микрофон {i}':<12} {port:<8} {'⚠️ Нет данных':<20} {'':<12} {'':<15} {0:<15}")
                
                # Выводим информацию о секторе и координатах
                sector_color = "\033[95m" if current_sector in ["СВЕРХУ-СПРАВА", "СНИЗУ", "СВЕРХУ-СЛЕВА"] else \
                             "\033[91m" if current_sector == "ОШИБКА РАСПОЗНОВАНИЯ" else "\033[92m"
                
                if localization_result["coordinates"]:
                    coords = localization_result["coordinates"]
                    print(f"\n{sector_color}Определен сектор: {current_sector} | Координаты: x={coords['x']:.2f} м, y={coords['y']:.2f} м{reset}")
                else:
                    print(f"\n{sector_color}Определен сектор: {current_sector}{reset}")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\n⚠️ Ошибка обработки: {str(e)}")
                time.sleep(1)
                
        print("\n❌ Прерывание обработки аудио.")

    def determine_sector(self):
        """Определение сектора обнаружения на основе уровней сигнала с разных микрофонов"""
        # Словарь соответствия портов микрофонам
        MICROPHONE_MAP = {
            5000: "M1",  # Центральный микрофон
            5001: "M2",  # Верхний микрофон
            5002: "M3",  # Левый нижний микрофон
            5003: "M4"   # Правый нижний микрофон
        }
        
        # Получаем последние предсказания
        predictions = self.last_predictions
        print(f"Текущие предсказания: {predictions}")  # Отладочная информация
        
        # Обновляем предсказания в локализаторе
        self.localizer.update_predictions(predictions)
        
        # Проверяем наличие всех необходимых устройств
        if not all(port in predictions for port in PORTS):
            missing_ports = [port for port in PORTS if port not in predictions]
            print(f"Отсутствуют данные с портов: {missing_ports}")  # Отладочная информация
            return {
                "sector": "Не хватает данных с микрофонов",
                "coordinates": None,
                "error": "Недостаточно данных"
            }
        
        # Собираем информацию о распознанных дронах
        detected_mics = []
        sound_levels = {}
        
        for port in PORTS:
            pred = predictions[port]
            if pred[0] == "class1":  # Если обнаружен дрон
                detected_mics.append(port)
                sound_levels[port] = pred[2]  # Сохраняем уровень звука
        
        # Если не обнаружено ни одного дрона
        if not detected_mics:
            return {
                "sector": "Фоновый шум",
                "coordinates": None,
                "error": None
            }
        
        # Вычисляем TDOA для каждой пары микрофонов
        tdoas = {}
        for port in detected_mics:
            if port != 5000:  # Пропускаем центральный микрофон
                mic_name = MICROPHONE_MAP[port]
                # Вычисляем TDOA относительно центрального микрофона
                tdoa = calculate_tdoa(
                    self.audio_buffers[5000],
                    self.audio_buffers[port],
                    TARGET_SAMPLE_RATE
                )
                tdoas[mic_name] = tdoa
        
        try:
            # Определяем координаты дрона
            drone_pos = self.localizer.localize(tdoas)
            
            # Определяем сектор
            sector = self.localizer.get_sector(drone_pos)
            
            # Формируем результат
            result = {
                "sector": sector,
                "coordinates": {
                    "x": float(drone_pos[0]),
                    "y": float(drone_pos[1])
                },
                "error": None
            }
            
            return result
            
        except ValueError as e:
            print(f"Ошибка локализации: {str(e)}")
            return {
                "sector": "ОШИБКА РАСПОЗНОВАНИЯ",
                "coordinates": None,
                "error": str(e)
            }
        
        # Если локализация не удалась, используем старый метод определения сектора
        if len(detected_mics) == 4:
            min_level_port = min(sound_levels, key=sound_levels.get)
            detected_mics.remove(min_level_port)
            print(f"Исключен микрофон {MICROPHONE_MAP[min_level_port]} с уровнем {sound_levels[min_level_port]:.1f} дБ")
        
        if 5000 not in detected_mics:
            return {
                "sector": "ОШИБКА РАСПОЗНОВАНИЯ",
                "coordinates": None,
                "error": "Не обнаружен центральный микрофон"
            }
        
        detected_mics_set = set(detected_mics)
        
        if detected_mics_set == {5000, 5001, 5002}:
            return {
                "sector": "СВЕРХУ-СПРАВА",
                "coordinates": None,
                "error": "Используется приближенное определение сектора"
            }
        
        if detected_mics_set == {5000, 5002, 5003}:
            return {
                "sector": "СНИЗУ",
                "coordinates": None,
                "error": "Используется приближенное определение сектора"
            }
        
        if detected_mics_set == {5000, 5001, 5003}:
            return {
                "sector": "СВЕРХУ-СЛЕВА",
                "coordinates": None,
                "error": "Используется приближенное определение сектора"
            }
        
        return {
            "sector": "ОШИБКА РАСПОЗНОВАНИЯ",
            "coordinates": None,
            "error": "Неизвестная комбинация микрофонов"
        }

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