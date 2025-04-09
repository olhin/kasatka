import pyaudio
import numpy as np
import socket
import struct
import time
import argparse
import sys
import wave
import os



# Настройки аудио
CHUNK = 1024  # количество семплов в буфере
FORMAT = pyaudio.paFloat32  # формат аудио
CHANNELS = 1  # моно
RATE = 44100  # частота дискретизации
DEVICE_INDEX = None  # индекс устройства (None - устройство по умолчанию)

# Настройки сети
SERVER_IP = "127.0.0.1"  # IP-адрес сервера (локальный по умолчанию)
PORT_BASE = 5000  # базовый порт, к которому добавляется номер микрофона

def list_audio_devices():
    """Вывод доступных аудиоустройств"""
    p = pyaudio.PyAudio()
    
    print("\n=== Доступные аудиоустройства ===")
    print("Индекс\tНазвание\t(Каналы, Частота)")
    print("-" * 50)
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info["name"]
        channels = info["maxInputChannels"]
        rate = int(info["defaultSampleRate"])
        
        if channels > 0:  # Только устройства ввода
            print(f"{i}\t{name}\t({channels} ch, {rate} Hz)")
    
    print("-" * 50)
    p.terminate()

def capture_and_send(device_index, mic_number, server_ip, test_file=None):
    """Захват аудиоданных и отправка по UDP"""
    port = PORT_BASE + mic_number - 1  # Порты 5000, 5001, 5002, 5003
    
    # Создаем UDP сокет
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Инициализация аудио
    p = pyaudio.PyAudio()
    
    # Получение информации об устройстве
    if device_index is not None:
        device_info = p.get_device_info_by_index(device_index)
        print(f"Использование устройства: {device_info['name']}")
    
    # Для тестирования - чтение из файла, а не с микрофона
    if test_file:
        return send_from_wavfile(test_file, sock, server_ip, port)
    
    # Открытие потока аудио
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK
    )
    
    print(f"Запущен захват с микрофона {mic_number} (устройство {device_index if device_index is not None else 'по умолчанию'})")
    print(f"Данные отправляются на {server_ip}:{port}")
    print(f"Нажмите Ctrl+C для остановки")
    
    try:
        while True:
            # Захват данных
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Преобразование в numpy массив
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            # Текущее время для временной метки
            timestamp = time.time()
            
            # Формирование пакета: [timestamp(8 байт)] + [audio_data]
            packet = struct.pack('d', timestamp) + audio_chunk.tobytes()
            
            # Отправка по UDP
            sock.sendto(packet, (server_ip, port))
            
    except KeyboardInterrupt:
        print(f"\nОстановка записи с микрофона {mic_number}")
    except Exception as e:
        print(f"Ошибка: {str(e)}")
    finally:
        # Закрытие потока и освобождение ресурсов
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()

def send_from_wavfile(filename, sock, server_ip, port):
    """Отправка данных из WAV-файла вместо микрофона (для тестирования)"""
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден!")
        return False
    
    try:
        wf = wave.open(filename, 'rb')
        
        # Проверка параметров файла
        if wf.getnchannels() != CHANNELS:
            print(f"Предупреждение: WAV-файл имеет {wf.getnchannels()} каналов вместо {CHANNELS}")
        
        if wf.getframerate() != RATE:
            print(f"Предупреждение: WAV-файл имеет частоту {wf.getframerate()} Hz вместо {RATE} Hz")
        
        print(f"Отправка данных из файла {filename} на {server_ip}:{port}")
        print(f"Формат файла: {wf.getnchannels()} каналов, {wf.getframerate()} Hz")
        
        # Имитация потоковой передачи файла
        data = wf.readframes(CHUNK)
        
        while data:
            # Преобразование в numpy массив
            if wf.getsampwidth() == 4:  # 32-bit float
                audio_chunk = np.frombuffer(data, dtype=np.float32)
            elif wf.getsampwidth() == 2:  # 16-bit int
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                print(f"Неподдерживаемая битность {wf.getsampwidth() * 8} бит")
                break
                
            # Текущее время для временной метки
            timestamp = time.time()
            
            # Формирование пакета
            packet = struct.pack('d', timestamp) + audio_chunk.tobytes()
            
            # Отправка по UDP
            sock.sendto(packet, (server_ip, port))
            
            # Задержка, примерно соответствующая потоковой передаче
            time.sleep(CHUNK / RATE)
            
            # Чтение следующего чанка
            data = wf.readframes(CHUNK)
        
        print(f"Отправка файла завершена")
        return True
        
    except Exception as e:
        print(f"Ошибка при отправке файла: {str(e)}")
        return False
    finally:
        if 'wf' in locals():
            wf.close()
        sock.close()

def main():
    parser = argparse.ArgumentParser(description='Захват аудио с микрофона и отправка по UDP')
    parser.add_argument('--list', action='store_true', help='Вывести список доступных аудиоустройств')
    parser.add_argument('--device', type=int, help='Индекс устройства записи')
    parser.add_argument('--mic', type=int, choices=[1, 2, 3, 4], default=1, 
                        help='Номер микрофона (1-4, влияет на порт отправки)')
    parser.add_argument('--server', type=str, default=SERVER_IP, help='IP-адрес сервера')
    parser.add_argument('--test', type=str, help='Путь к WAV-файлу для тестирования')
    
    args = parser.parse_args()
    
    if args.list:
        list_audio_devices()
        return
    
    capture_and_send(args.device, args.mic, args.server, args.test)

if __name__ == "__main__":
    main() 