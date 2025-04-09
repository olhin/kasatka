@echo off
echo Запуск системы обнаружения звуков...
echo ===================================
echo.

:: Задаем переменные для путей
set BACKEND_PATH=%~dp0
set FRONTEND_PATH=%~dp0locator_static

:: Выбор режима работы
echo Выберите режим работы:
echo 1. Обнаружение дронов (стандартный режим)
echo 2. Демо-режим с распознаванием хлопков
echo.
set /p MODE_CHOICE=Введите номер (1 или 2): 

:: Устанавливаем параметры режима
set MODE_PARAM=
if "%MODE_CHOICE%"=="2" (
    set MODE_PARAM=--mode clap
    echo.
    echo ЗАПУСК В ДЕМО-РЕЖИМЕ С РАСПОЗНАВАНИЕМ ХЛОПКОВ
) else (
    echo.
    echo ЗАПУСК В СТАНДАРТНОМ РЕЖИМЕ ОБНАРУЖЕНИЯ ДРОНОВ
)

:: Запуск бэкенда (Python скрипт)
echo.
echo Запуск нейросети (RasbaryPi2.py)...
start cmd /k "cd %BACKEND_PATH% && python RasbaryPi2.py %MODE_PARAM%"

:: Ожидание запуска бэкенда
echo Ожидание запуска WebSocket сервера...
ping -n 5 127.0.0.1 > nul

:: Запуск фронтенда (React приложение)
echo Запуск фронтенда...
start cmd /k "cd %FRONTEND_PATH% && npm start"

echo.

echo Система запущена. Оба терминала должны оставаться открытыми.
echo Фронтенд будет доступен по адресу http://localhost:3000 