@echo off
echo Копирование файлов модели для режима с хлопками
echo ================================================
echo.

set SCRIPT_PATH=%~dp0
set PARENT_PATH=%SCRIPT_PATH%..

echo Поиск файлов моделей...

set TFLITE_FILE=soundclassifier_with_metadata.tflite
set LABELS_FILE=labels.txt

:: Проверка существования файлов в текущей директории
if exist "%SCRIPT_PATH%%TFLITE_FILE%" (
    echo Файл %TFLITE_FILE% уже существует в текущей директории.
) else (
    :: Проверка существования файлов в родительской директории
    if exist "%PARENT_PATH%\%TFLITE_FILE%" (
        echo Копирование %TFLITE_FILE% из родительской директории...
        copy "%PARENT_PATH%\%TFLITE_FILE%" "%SCRIPT_PATH%%TFLITE_FILE%" >nul
        echo Файл скопирован успешно!
    ) else (
        echo ВНИМАНИЕ: Файл %TFLITE_FILE% не найден! 
        echo Поместите этот файл в директорию %SCRIPT_PATH% или %PARENT_PATH%\
    )
)

:: Проверка существования файлов меток в текущей директории
if exist "%SCRIPT_PATH%%LABELS_FILE%" (
    echo Файл %LABELS_FILE% уже существует в текущей директории.
) else (
    :: Проверка существования файлов в родительской директории
    if exist "%PARENT_PATH%\%LABELS_FILE%" (
        echo Копирование %LABELS_FILE% из родительской директории...
        copy "%PARENT_PATH%\%LABELS_FILE%" "%SCRIPT_PATH%%LABELS_FILE%" >nul
        echo Файл скопирован успешно!
    ) else (
        echo ВНИМАНИЕ: Файл %LABELS_FILE% не найден!
        echo Поместите этот файл в директорию %SCRIPT_PATH% или %PARENT_PATH%\
    )
)

echo.
echo Операция завершена. Теперь вы можете запустить демо-режим с распознаванием хлопков.
echo Запустите clap_startup.bat для запуска демо-режима.
echo.
pause 