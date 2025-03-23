@echo off
echo Запуск системы обнаружения дронов...
echo.

:: Задаем переменные для путей
set BACKEND_PATH=%~dp0
set FRONTEND_PATH=%~dp0locator_static

:: Запуск бэкенда (Python скрипт)
echo Запуск нейросети (RasbaryPi2.py)...
start cmd /k "cd %BACKEND_PATH% && python RasbaryPi2.py"

:: Ожидание запуска бэкенда
echo Ожидание запуска WebSocket сервера...
ping -n 5 127.0.0.1 > nul

:: Запуск фронтенда (React приложение)
echo Запуск фронтенда...
start cmd /k "cd %FRONTEND_PATH% && npm start"

echo.

echo Система запущена. Оба терминала должны оставаться открытыми.
echo Фронтенд будет доступен по адресу http://localhost:3000 