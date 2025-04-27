import numpy as np
from scipy.optimize import minimize
import math

class DroneLocalizer:
    def __init__(self, speed_of_sound=343.0):
        # Координаты микрофонов
        self.microphones = {
            'M1': np.array([0, 0]),      # Центральный микрофон
            'M2': np.array([0, 7]),      # Верхний микрофон
            'M3': np.array([-3.5*math.sqrt(3), -3.5]),  # Левый нижний микрофон
            'M4': np.array([3.5*math.sqrt(3), -3.5])    # Правый нижний микрофон
        }
        self.v = speed_of_sound  # Скорость звука в м/с
        
        # Порты для микрофонов
        self.PORTS = [5000, 5001, 5002, 5003]
        
        # Словарь соответствия портов микрофонам
        self.MICROPHONE_MAP = {
            5000: "M1",  # Центральный микрофон
            5001: "M2",  # Верхний микрофон
            5002: "M3",  # Левый нижний микрофон
            5003: "M4"   # Правый нижний микрофон
        }
        
        # Хранилище последних предсказаний
        self.last_predictions = {}

    def calculate_distance_difference(self, drone_pos, mic1, mic2):
        """Вычисляет разницу расстояний от дрона до двух микрофонов"""
        d1 = np.linalg.norm(drone_pos - mic1)
        d2 = np.linalg.norm(drone_pos - mic2)
        return d2 - d1

    def error_function(self, pos, tdoas):
        """Функция ошибки для оптимизации"""
        pos = np.array(pos)
        error = 0.0
        
        # Вычисляем ошибку для каждой пары микрофонов
        for mic_name, tdoa in tdoas.items():
            mic_pos = self.microphones[mic_name]
            # Вычисляем расстояние от дрона до каждого микрофона
            distance = np.linalg.norm(pos - mic_pos)
            # Вычисляем ожидаемую разницу расстояний
            expected_delta = tdoa * self.v
            # Вычисляем фактическую разницу расстояний относительно центрального микрофона
            actual_delta = distance - np.linalg.norm(pos - self.microphones['M1'])
            error += (expected_delta - actual_delta)**2
            
        return error

    def localize(self, tdoas):
        """
        Определяет координаты дрона на основе TDOA
        
        Args:
            tdoas (dict): Словарь с задержками в формате {'M2': Δt₂, 'M3': Δt₃, 'M4': Δt₄}
            
        Returns:
            tuple: (x, y) координаты дрона
        """
        # Начальное предположение - центр системы координат
        initial_guess = np.array([0, 0])
        
        # Минимизация функции ошибки
        result = minimize(
            self.error_function,
            initial_guess,
            args=(tdoas,),
            method='L-BFGS-B',
            bounds=[(-100, 100), (-100, 100)]  # Увеличиваем область поиска
        )
        
        if result.success:
            return result.x
        else:
            raise ValueError("Не удалось определить местоположение дрона")

    def get_sector(self, drone_pos):
        """
        Определяет сектор, в котором находится дрон
        
        Args:
            drone_pos (np.array): Координаты дрона (x, y)
            
        Returns:
            str: Название сектора
        """
        x, y = drone_pos
        
        # Определяем угол относительно центрального микрофона
        angle = np.arctan2(y, x) * 180 / np.pi
        
        # Корректируем угол для удобства определения сектора
        if angle < 0:
            angle += 360
            
        # Определяем сектор на основе угла
        if 0 <= angle < 120:
            return "СВЕРХУ-СПРАВА"
        elif 120 <= angle < 240:
            return "СНИЗУ"
        elif 240 <= angle < 360:
            return "СВЕРХУ-СЛЕВА"
        else:
            return "ОШИБКА РАСПОЗНОВАНИЯ"

    def update_predictions(self, predictions):
        """
        Обновляет последние предсказания для всех микрофонов
        
        Args:
            predictions (dict): Словарь с предсказаниями в формате {port: prediction}
        """
        self.last_predictions = predictions
        print(f"Обновлены предсказания: {predictions}")  # Для отладки

def calculate_tdoa(audio_data1, audio_data2, sample_rate):
    """
    Вычисляет разницу времени прихода сигнала (TDOA) между двумя микрофонами
    
    Args:
        audio_data1 (np.array): Аудиоданные с первого микрофона
        audio_data2 (np.array): Аудиоданные со второго микрофона
        sample_rate (int): Частота дискретизации
        
    Returns:
        float: TDOA в секундах
    """
    # Вычисляем взаимную корреляцию
    correlation = np.correlate(audio_data1, audio_data2, mode='full')
    
    # Находим индекс максимальной корреляции
    max_index = np.argmax(correlation)
    
    # Преобразуем индекс в задержку в секундах
    delay = (max_index - len(audio_data1) + 1) / sample_rate
    
    return delay 