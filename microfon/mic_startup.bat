@echo off
echo Запуск системы обнаружения дронов с микрофонами...
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

:: Проверка зависимостей
echo Проверка необходимых библиотек...
python -c "import numpy, socket, struct, time, wave" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Установка зависимостей...
    pip install numpy
)

:: Задаем переменные для путей
set SCRIPT_PATH=%~dp0
set SERVER_IP=127.0.0.1

:: Показать доступные аудиоустройства
echo Доступные аудиоустройства:
python "%SCRIPT_PATH%mic_sender.py" --list

echo.
echo ВАЖНО: По умолчанию система будет использовать одно и то же аудиоустройство 
echo для имитации 4 разных микрофонов. В реальном сценарии вы можете подключить
echo разные микрофоны, указав их индексы.
echo.

:: Запрос у пользователя, какое устройство использовать
set /p DEVICE_INDEX=Введите индекс устройства для использования (или нажмите Enter для устройства по умолчанию): 

:: Запуск основного сервера
echo.
echo Запуск основного сервера (RasbaryPi2.py)...
start cmd /k "cd %SCRIPT_PATH% && python RasbaryPi2.py"

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

:: Запуск тестовых WAV-файлов, если они есть
set TEST_DIR=%SCRIPT_PATH%test
set MIC1_TEST=
set MIC2_TEST=
set MIC3_TEST=
set MIC4_TEST=

:: Проверка наличия тестовых файлов
if exist "%TEST_DIR%\mic1.wav" set MIC1_TEST=--test "%TEST_DIR%\mic1.wav"
if exist "%TEST_DIR%\mic2.wav" set MIC2_TEST=--test "%TEST_DIR%\mic2.wav"
if exist "%TEST_DIR%\mic3.wav" set MIC3_TEST=--test "%TEST_DIR%\mic3.wav"
if exist "%TEST_DIR%\mic4.wav" set MIC4_TEST=--test "%TEST_DIR%\mic4.wav"

:: Запрос у пользователя, использовать тестовые файлы или микрофоны
echo.
set /p USE_TEST=Использовать тестовые файлы для имитации микрофонов? (y/n, по умолчанию - n): 

:: Запуск микрофонов или тестовых файлов
if /i "%USE_TEST%"=="y" (
    :: Проверка наличия тестовой директории
    if not exist "%TEST_DIR%" (
        mkdir "%TEST_DIR%"
        echo Создана директория %TEST_DIR% для тестовых файлов.
        echo Пожалуйста, поместите WAV-файлы mic1.wav, mic2.wav, mic3.wav, mic4.wav в эту директорию.
        echo Затем запустите скрипт снова.
        pause
        exit /b 0
    )
    
    :: Запуск с тестовыми файлами, если они есть
    if defined MIC1_TEST (
        start cmd /k "cd %SCRIPT_PATH% && python mic_sender.py --mic 1 --server %SERVER_IP% %MIC1_TEST%"
    ) else (
        echo Файл mic1.wav не найден в %TEST_DIR%
    )
    
    if defined MIC2_TEST (
        start cmd /k "cd %SCRIPT_PATH% && python mic_sender.py --mic 2 --server %SERVER_IP% %MIC2_TEST%"
    ) else (
        echo Файл mic2.wav не найден в %TEST_DIR%
    )
    
    if defined MIC3_TEST (
        start cmd /k "cd %SCRIPT_PATH% && python mic_sender.py --mic 3 --server %SERVER_IP% %MIC3_TEST%"
    ) else (
        echo Файл mic3.wav не найден в %TEST_DIR%
    )
    
    if defined MIC4_TEST (
        start cmd /k "cd %SCRIPT_PATH% && python mic_sender.py --mic 4 --server %SERVER_IP% %MIC4_TEST%"
    ) else (
        echo Файл mic4.wav не найден в %TEST_DIR%
    )
) else (
    :: Запуск с реальными микрофонами
    start cmd /k "cd %SCRIPT_PATH% && python mic_sender.py --mic 1 --server %SERVER_IP% %DEVICE_PARAM%"
    timeout /t 1 >nul
    start cmd /k "cd %SCRIPT_PATH% && python mic_sender.py --mic 2 --server %SERVER_IP% %DEVICE_PARAM%"
    timeout /t 1 >nul
    start cmd /k "cd %SCRIPT_PATH% && python mic_sender.py --mic 3 --server %SERVER_IP% %DEVICE_PARAM%"
    timeout /t 1 >nul
    start cmd /k "cd %SCRIPT_PATH% && python mic_sender.py --mic 4 --server %SERVER_IP% %DEVICE_PARAM%"
)

:: Запуск фронтенда
echo.
echo Запуск фронтенда...
start cmd /k "cd %SCRIPT_PATH%\locator_static && npm start"

echo.
echo Система запущена!
echo Для остановки закройте все открытые окна командной строки.
echo. 