@echo off
echo Установка зависимостей для системы обнаружения звуков
echo =======================================================
echo.

:: Проверка наличия Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: Python не найден. Пожалуйста, установите Python 3.
    pause
    exit /b 1
)

:: Установка основных библиотек Python
echo Установка основных библиотек Python...
pip install torch numpy librosa scikit-learn websockets asyncio pyaudio

:: Установка TensorFlow
echo.
echo Установка TensorFlow для демо-режима...
pip install tensorflow

:: Проверка успешности установки
python -c "import tensorflow as tf; print('TensorFlow версия:', tf.__version__)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: Не удалось установить TensorFlow. Демо-режим не будет работать.
    echo Попробуйте установить вручную: pip install tensorflow
) else (
    echo TensorFlow успешно установлен.
)

:: Переход в директорию фронтенда и установка npm-пакетов
echo.
echo Установка зависимостей для фронтенда...
cd locator_static
call npm install

:: Проверка успешности установки npm-пакетов
if %ERRORLEVEL% NEQ 0 (
    echo Ошибка при установке npm-пакетов!
    cd ..
    pause
    exit /b 1
)
echo Зависимости для фронтенда успешно установлены.
echo.

:: Возврат в исходную директорию
cd ..

echo Установка завершена успешно!
echo Теперь вы можете запустить систему с помощью startup.bat
echo.
pause 