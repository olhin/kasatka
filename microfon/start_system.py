#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import time
import signal

def run_backend():
    """Запуск бэкенда - скрипта с нейросетью"""
    print("Запуск бэкенда (RasbaryPi2.py)...")
    return subprocess.Popen(
        ['python', 'RasbaryPi2.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

def run_frontend():
    """Запуск фронтенда - React-приложения"""
    frontend_dir = os.path.join(os.getcwd(), 'locator  static')
    print(f"Запуск фронтенда из директории: {frontend_dir}")
    
    # Проверка, существует ли директория
    if not os.path.exists(frontend_dir):
        print(f"Ошибка: директория '{frontend_dir}' не найдена!")
        return None
    
    return subprocess.Popen(
        ['npm', 'start'],
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

def print_process_output(process, prefix):
    """Вывод логов процесса в реальном времени"""
    if process and process.stdout:
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            print(f"{prefix}: {line.strip()}")

def main():
    """Основная функция запуска системы"""
    print("="*60)
    print("Запуск системы мониторинга дронов")
    print("="*60)
    
    try:
        # Запуск бэкенда
        backend_process = run_backend()
        if not backend_process:
            print("Ошибка запуска бэкенда!")
            return 1
            
        # Даем время на инициализацию бэкенда
        time.sleep(3)
        
        # Запуск фронтенда
        frontend_process = run_frontend()
        if not frontend_process:
            print("Ошибка запуска фронтенда!")
            backend_process.terminate()
            return 1
        
        # Мониторинг процессов
        print("Система запущена успешно.")
        print("Нажмите Ctrl+C для завершения...")
        
        # Цикл для вывода логов в реальном времени
        try:
            while True:
                # Проверка, что процессы еще работают
                if backend_process.poll() is not None:
                    print("Процесс бэкенда завершился!")
                    break
                    
                if frontend_process.poll() is not None:
                    print("Процесс фронтенда завершился!")
                    break
                
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nПолучен сигнал завершения...")
        
    except KeyboardInterrupt:
        print("\nПрерывание от пользователя...")
    finally:
        # Корректное завершение процессов
        print("Завершение работы системы...")
        
        if 'backend_process' in locals() and backend_process:
            backend_process.terminate()
            print("Бэкенд остановлен")
            
        if 'frontend_process' in locals() and frontend_process:
            frontend_process.terminate()
            print("Фронтенд остановлен")
            
    return 0

if __name__ == "__main__":
    sys.exit(main()) 