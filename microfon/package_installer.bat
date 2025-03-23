@echo off
echo Установка зависимостей для системы обнаружения дронов
echo =======================================================
echo.

:: Установка библиотек Python
echo Установка библиотек Python...
pip install torch numpy librosa scikit-learn websockets asyncio

:: Проверка успешности установки Python-библиотек
if %ERRORLEVEL% NEQ 0 (
    echo Ошибка при установке Python-библиотек!
    pause
    exit /b 1
)
echo Python-библиотеки успешно установлены.
echo.

:: Переход в директорию фронтенда и установка npm-пакетов
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