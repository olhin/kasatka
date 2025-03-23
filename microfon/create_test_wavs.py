"""
Скрипт для создания тестовых WAV-файлов для имитации микрофонов

Создает 4 файла с синусоидальными сигналами разной частоты и белым шумом,
которые можно использовать для тестирования системы обнаружения дронов
без наличия реальных микрофонов.
"""

import numpy as np
import wave
import struct
import os

# Параметры аудио
SAMPLE_RATE = 44100  # Частота дискретизации
DURATION = 10  # Длительность в секундах
AMPLITUDE = 0.3  # Амплитуда сигнала (от 0 до 1)
NOISE_LEVEL = 0.05  # Уровень шума (от 0 до 1)

# Параметры для сигналов дрона (Hz)
DRONE_FREQUENCIES = [200, 300, 400, 500]  # Разные частоты для каждого микрофона

# Директория для сохранения
OUTPUT_DIR = "test"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_drone_sound(freq, duration, sample_rate, amplitude, noise_level):
    """Генерирует сигнал, имитирующий звук дрона"""
    # Создаем таймлайн
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Основная частота с гармониками
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    signal += amplitude * 0.5 * np.sin(2 * np.pi * (freq*2) * t)
    signal += amplitude * 0.25 * np.sin(2 * np.pi * (freq*3) * t)
    
    # Добавляем модуляцию амплитуды (пульсацию)
    modulation = 1 + 0.2 * np.sin(2 * np.pi * 5 * t)  # 5 Hz модуляция
    signal = signal * modulation
    
    # Добавляем случайный шум
    noise = np.random.normal(0, 1, len(t)) * noise_level
    signal = signal + noise
    
    # Нормализация для предотвращения клиппинга
    signal = signal / np.max(np.abs(signal))
    
    return signal

def save_wav_file(filename, signal, sample_rate):
    """Сохраняет сигнал в WAV-файл"""
    # Преобразуем в диапазон -32767 до 32767
    audio_data = np.int16(signal * 32767)
    
    with wave.open(filename, 'w') as wav_file:
        # Настройка параметров WAV-файла
        wav_file.setnchannels(1)  # Моно
        wav_file.setsampwidth(2)  # 16 бит
        wav_file.setframerate(sample_rate)
        
        # Запись данных
        for sample in audio_data:
            # Упаковываем в 16-битный формат
            packed_value = struct.pack('h', sample)
            wav_file.writeframes(packed_value)
    
    return len(audio_data)

def create_test_files():
    """Создает тестовые WAV-файлы для всех микрофонов"""
    print("Создание тестовых WAV-файлов...")
    
    for i, freq in enumerate(DRONE_FREQUENCIES, 1):
        # Генерируем сигнал
        signal = generate_drone_sound(
            freq=freq,
            duration=DURATION,
            sample_rate=SAMPLE_RATE,
            amplitude=AMPLITUDE,
            noise_level=NOISE_LEVEL
        )
        
        # Имя файла
        filename = os.path.join(OUTPUT_DIR, f"mic{i}.wav")
        
        # Сохраняем файл
        samples = save_wav_file(filename, signal, SAMPLE_RATE)
        
        print(f"Создан файл {filename}")
        print(f"  - Частота сигнала: {freq} Hz")
        print(f"  - Длительность: {DURATION} сек")
        print(f"  - Частота дискретизации: {SAMPLE_RATE} Hz")
        print(f"  - Размер: {samples} сэмплов\n")

def create_class_files():
    """Создает тестовые WAV-файлы для разных классов звуков"""
    print("Создание файлов для обучения нейросети...")
    
    # Создаем директории для классов
    for class_dir in ['train/class1', 'train/class2', 'valid/class1', 'valid/class2']:
        full_path = os.path.join(os.path.dirname(OUTPUT_DIR), class_dir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
    
    # Класс 1: дрон (разные звуковые характеристики)
    drone_frequencies = [220, 240, 260, 280, 300, 320, 340]
    
    for i, freq in enumerate(drone_frequencies):
        # Генерируем сигнал дрона
        signal = generate_drone_sound(
            freq=freq,
            duration=2,  # 2-секундные файлы
            sample_rate=SAMPLE_RATE,
            amplitude=AMPLITUDE,
            noise_level=NOISE_LEVEL * 0.8  # Меньше шума для лучшего обучения
        )
        
        # Сохраняем в train и valid
        save_wav_file(
            os.path.join(os.path.dirname(OUTPUT_DIR), 'train/class1', f"drone_{i}.wav"), 
            signal, 
            SAMPLE_RATE
        )
        
        # Слегка модифицируем для валидации
        signal_mod = signal + np.random.normal(0, 0.01, len(signal))
        save_wav_file(
            os.path.join(os.path.dirname(OUTPUT_DIR), 'valid/class1', f"drone_{i}.wav"), 
            signal_mod, 
            SAMPLE_RATE
        )
    
    # Класс 2: фоновый шум (больше шума, меньше сигнала)
    for i in range(7):
        # Генерируем шум с очень слабым сигналом
        t = np.linspace(0, 2, int(SAMPLE_RATE * 2), False)
        noise = np.random.normal(0, 1, len(t)) * 0.5
        weak_signal = np.sin(2 * np.pi * 100 * t) * 0.05  # Очень слабый сигнал
        
        signal = noise + weak_signal
        signal = signal / np.max(np.abs(signal))
        
        # Сохраняем в train и valid
        save_wav_file(
            os.path.join(os.path.dirname(OUTPUT_DIR), 'train/class2', f"noise_{i}.wav"), 
            signal, 
            SAMPLE_RATE
        )
        
        # Слегка модифицируем для валидации
        signal_mod = signal + np.random.normal(0, 0.01, len(signal))
        save_wav_file(
            os.path.join(os.path.dirname(OUTPUT_DIR), 'valid/class2', f"noise_{i}.wav"), 
            signal_mod, 
            SAMPLE_RATE
        )
    
    print(f"Созданы файлы для обучения в директориях train/ и valid/")

if __name__ == "__main__":
    # Создаем тестовые файлы для микрофонов
    create_test_files()
    
    # Спрашиваем, нужно ли создать файлы для обучения
    create_train_files = input("Создать файлы для обучения нейросети? (y/n): ")
    if create_train_files.lower() == 'y':
        create_class_files()
    
    print("\nГотово! Файлы созданы. Используйте их для тестирования системы.")
    print("Для запуска тестирования используйте mic_startup.bat и выберите опцию использования тестовых файлов.") 