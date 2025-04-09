@echo off
echo Запуск системы обнаружения хлопков (ДЕМО-РЕЖИМ)...
echo ===================================================
echo.

:: Проверка наличия Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: Python не найден. Пожалуйста, установите Python 3.
    pause
    exit /b 1
)

:: Проверка наличия PyAudio
python -c "import pyaudio" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Установка PyAudio...
    pip install pyaudio
    if %ERRORLEVEL% NEQ 0 (
        echo ОШИБКА: Не удалось установить PyAudio
        echo Попробуйте установить вручную: pip install pyaudio
        pause
        exit /b 1
    )
)

:: Проверка TensorFlow
python -c "import tensorflow as tf" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ВНИМАНИЕ: TensorFlow не установлен. Демо-режим не будет работать.
    echo Выполняется установка TensorFlow...
    pip install tensorflow
    if %ERRORLEVEL% NEQ 0 (
        echo ОШИБКА: Не удалось установить TensorFlow
        echo Вы можете запустить install_dependencies.bat для установки всех зависимостей
        echo или установить вручную: pip install tensorflow
        pause
        exit /b 1
    )
)

:: Задаем переменные для путей
set SCRIPT_PATH=%~dp0
set SERVER_IP=127.0.0.1

:: Копирование файлов модели
echo.
echo Проверка и копирование файлов модели...
call "%SCRIPT_PATH%copy_model_files.bat"

:: Проверка зависимостей
echo Проверка необходимых библиотек...
python -c "import numpy, socket, struct, time, wave" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Установка зависимостей...
    pip install numpy
)

:: Показать доступные аудиоустройства
echo Доступные аудиоустройства:
python "%SCRIPT_PATH%mic_sender.py" --list

echo.
echo ВАЖНО: Для демонстрации хлопков вам понадобится реальный микрофон. Выберите 
echo устройство с номером, соответствующим вашему микрофону.
echo.

:: Запрос у пользователя, какое устройство использовать
set /p DEVICE_INDEX=Введите индекс устройства для использования (или нажмите Enter для устройства по умолчанию): 

:: Запуск основного сервера в режиме хлопков
echo.
echo Запуск основного сервера в режиме хлопков (RasbaryPi2.py)...
start cmd /k "cd %SCRIPT_PATH% && python RasbaryPi2.py --mode clap"

:: Пауза для запуска сервера
echo Ожидание запуска сервера...
timeout /t 3 >nul

:: Запуск микрофонных клиентов
echo Запуск микрофонных клиентов...

:: Параметр устройства, если указан
set DEVICE_PARAM=
if not "%DEVICE_INDEX%"=="" (
    set DEVICE_PARAM=--device %DEVICE_INDEX%
)

:: Запуск с реальными микрофонами
start cmd /k "cd %SCRIPT_PATH% && python mic_sender.py --mic 1 --server %SERVER_IP% %DEVICE_PARAM%"
timeout /t 1 >nul
start cmd /k "cd %SCRIPT_PATH% && python mic_sender.py --mic 2 --server %SERVER_IP% %DEVICE_PARAM%"
timeout /t 1 >nul
start cmd /k "cd %SCRIPT_PATH% && python mic_sender.py --mic 3 --server %SERVER_IP% %DEVICE_PARAM%"
timeout /t 1 >nul
start cmd /k "cd %SCRIPT_PATH% && python mic_sender.py --mic 4 --server %SERVER_IP% %DEVICE_PARAM%"

:: Запуск фронтенда
echo.
echo Запуск фронтенда...
start cmd /k "cd %SCRIPT_PATH%\locator_static && npm start"

echo.
echo Демо-система распознавания хлопков запущена!
echo Сделайте хлопок возле микрофона, чтобы проверить работу.
echo Для остановки закройте все открытые окна командной строки.
echo. 