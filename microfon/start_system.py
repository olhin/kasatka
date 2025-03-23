#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import time
import signal
import shutil
import threading

def check_npm_exists():
    """Проверка наличия npm в системе"""
    npm_path = shutil.which('npm')
    if npm_path:
        print(f"npm найден: {npm_path}")
        return True
    else:
        print("ОШИБКА: npm не найден в системе!")
        print("Пожалуйста, установите Node.js с сайта https://nodejs.org/")
        return False

def run_backend():
    """Запуск бэкенда - скрипта с нейросетью"""
    print("Запуск бэкенда (RasbaryPi2.py)...")
    try:
        # Используем абсолютный путь к Python
        python_exe = sys.executable
        process = subprocess.Popen(
            [python_exe, 'RasbaryPi2.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Запустим поток для чтения вывода бэкенда
        threading.Thread(
            target=print_process_output,
            args=(process, "Бэкенд"),
            daemon=True
        ).start()
        
        return process
    except Exception as e:
        print(f"Ошибка запуска бэкенда: {str(e)}")
        return None

def run_frontend():
    """Запуск фронтенда - React-приложения"""
    # Используем нормализованный путь без пробелов
    frontend_dir = os.path.normpath(os.path.join(os.getcwd(), 'locator  static'))
    print(f"Запуск фронтенда из директории: {frontend_dir}")
    
    # Проверка, существует ли директория
    if not os.path.exists(frontend_dir):
        print(f"Ошибка: директория '{frontend_dir}' не найдена!")
        return None
    
    # Проверка наличия npm
    if not check_npm_exists():
        # Предлагаем альтернативу - открыть фронтенд в браузере
        print("Невозможно запустить фронтенд через npm.")
        print("Открываем браузерный интерфейс в режиме WebSocket-клиента...")
        
        # Создаем простой HTML файл для подключения к WebSocket
        html_path = os.path.join(os.getcwd(), 'websocket_client.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>WebSocket Клиент</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f0f0f0; }
        h1 { color: #333; }
        #status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .connected { background-color: #d4edda; color: #155724; }
        .disconnected { background-color: #f8d7da; color: #721c24; }
        #sector { font-size: 24px; font-weight: bold; margin: 20px 0; padding: 20px; background-color: #e9ecef; border-radius: 5px; }
        .sector-display { display: flex; margin: 20px 0; }
        .sector { width: 100px; height: 100px; margin: 10px; display: flex; justify-content: center; align-items: center; border-radius: 5px; background-color: #6c757d; color: white; }
        .active { background-color: #dc3545; animation: pulse 1s infinite; }
        @keyframes pulse { 0% { opacity: 0.7; } 50% { opacity: 1; } 100% { opacity: 0.7; } }
    </style>
</head>
<body>
    <h1>Монитор обнаружения дронов</h1>
    <div id="status" class="disconnected">Статус: Не подключено</div>
    <div id="sector">Сектор: Не определен</div>
    
    <div class="sector-display">
        <div class="sector" id="top-left">СВЕРХУ-СЛЕВА</div>
        <div class="sector" id="top-right">СВЕРХУ-СПРАВА</div>
        <div class="sector" id="bottom">СНИЗУ</div>
    </div>

    <script>
        const statusEl = document.getElementById('status');
        const sectorEl = document.getElementById('sector');
        const topLeft = document.getElementById('top-left');
        const topRight = document.getElementById('top-right');
        const bottom = document.getElementById('bottom');
        
        // Сбросить все активные секторы
        function resetSectors() {
            topLeft.classList.remove('active');
            topRight.classList.remove('active');
            bottom.classList.remove('active');
        }
        
        // Обработка сообщения от сервера
        function handleMessage(data) {
            if (data.sector) {
                sectorEl.textContent = `Сектор: ${data.sector}`;
                
                resetSectors();
                
                if (data.sector === 'СВЕРХУ-СЛЕВА') {
                    topLeft.classList.add('active');
                } else if (data.sector === 'СВЕРХУ-СПРАВА') {
                    topRight.classList.add('active');
                } else if (data.sector === 'СНИЗУ') {
                    bottom.classList.add('active');
                }
            }
        }
        
        // Подключение к WebSocket серверу
        function connect() {
            const socket = new WebSocket('ws://localhost:8765');
            
            socket.onopen = function() {
                statusEl.textContent = 'Статус: Подключено';
                statusEl.className = 'connected';
                console.log('WebSocket соединение установлено');
            };
            
            socket.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    console.log('Получены данные:', data);
                    handleMessage(data);
                } catch (error) {
                    console.error('Ошибка при обработке сообщения:', error);
                }
            };
            
            socket.onerror = function(error) {
                console.error('WebSocket ошибка:', error);
                statusEl.textContent = 'Статус: Ошибка соединения';
                statusEl.className = 'disconnected';
            };
            
            socket.onclose = function() {
                console.log('WebSocket соединение закрыто');
                statusEl.textContent = 'Статус: Отключено';
                statusEl.className = 'disconnected';
                
                // Пробуем переподключиться через 5 секунд
                setTimeout(connect, 5000);
            };
        }
        
        // Запускаем подключение при загрузке страницы
        connect();
    </script>
</body>
</html>
            """)
        
        # Пытаемся открыть HTML-файл в браузере
        try:
            import webbrowser
            print(f"Открываем резервный интерфейс в браузере: {html_path}")
            webbrowser.open('file://' + html_path)
            # Создаем фиктивный процесс для совместимости с основным кодом
            return subprocess.Popen(['echo', 'Фиктивный процесс фронтенда'], stdout=subprocess.PIPE)
        except Exception as e:
            print(f"Не удалось открыть браузер: {str(e)}")
            return None
    
    # Если npm доступен, запускаем через него
    try:
        return subprocess.Popen(
            ['npm', 'start'],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True  # Используем shell на Windows
        )
    except Exception as e:
        print(f"Ошибка запуска фронтенда: {str(e)}")
        return None

def print_process_output(process, prefix):
    """Вывод логов процесса в реальном времени"""
    if process and process.stdout:
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            print(f"{prefix} (stdout): {line.strip()}")
    
    # Также читаем stderr
    if process and process.stderr:
        for line in iter(process.stderr.readline, ''):
            if not line:
                break
            print(f"{prefix} (stderr): {line.strip()}")

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
        print("Ожидание инициализации бэкенда... (3 секунды)")
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
        
        # Цикл для мониторинга процессов
        try:
            while True:
                # Проверка, что процессы еще работают
                backend_status = backend_process.poll()
                if backend_status is not None:
                    print(f"Процесс бэкенда завершился с кодом {backend_status}!")
                    
                    # Вывод оставшейся информации из stderr
                    stderr_output, _ = backend_process.communicate()
                    if stderr_output:
                        print(f"Ошибка бэкенда: {stderr_output}")
                    break
                    
                frontend_status = frontend_process.poll() 
                if frontend_status is not None:
                    print(f"Процесс фронтенда завершился с кодом {frontend_status}!")
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
            try:
                backend_process.terminate()
                print("Бэкенд остановлен")
            except Exception as e:
                print(f"Ошибка при остановке бэкенда: {str(e)}")
            
        if 'frontend_process' in locals() and frontend_process:
            try:
                frontend_process.terminate()
                print("Фронтенд остановлен")
            except Exception as e:
                print(f"Ошибка при остановке фронтенда: {str(e)}")
            
    return 0

if __name__ == "__main__":
    sys.exit(main()) 