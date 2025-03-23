import React, { useState, useEffect, useCallback } from 'react';
import './App.css';

import Radar from './components/Radar/Radar';
import RulerRightLeft from './components/Ruler/RulerRightLeft';
import RulerTopBottom from './components/Ruler/RulerTopBottom';
import box from './Photo/eebc4b7dc05611ef8ab15ec00c958ca6_1.jpg';

let way = ""
let edinica = ' '
let object = ' '

// Настройка WebSocket соединения
const WS_URL = 'ws://localhost:8765'; // URL WebSocket сервера
const RECONNECT_INTERVAL = 3000; // Интервал переподключения в мс

function App() {
  const [showImage, setShowImage] = useState(false);
  const [isRed, setIsRed] = useState(false);
  const [isRed2, setIsRed2] = useState(false);
  const [isRed3, setIsRed3] = useState(false);
  const [currentSector, setCurrentSector] = useState("Не определен");
  const [wsConnected, setWsConnected] = useState(false);

  // Функция для отображения сектора в соответствии с данными от сервера
  const updateSectorDisplay = useCallback((sector) => {
    // Сначала сбрасываем все секторы
    setIsRed(false);
    setIsRed2(false);
    setIsRed3(false);

    // Затем включаем нужный сектор
    if (sector === 'СВЕРХУ-СЛЕВА') {
      setIsRed(true);
    } else if (sector === 'СВЕРХУ-СПРАВА') {
      setIsRed2(true);
    } else if (sector === 'СНИЗУ') {
      setIsRed3(true);
    }

    // Обновляем текущий сектор
    setCurrentSector(sector);
  }, []);

  // Создаем и управляем WebSocket соединением
  useEffect(() => {
    let socket = null;
    let reconnectTimer = null;

    // Функция для установки соединения
    const connect = () => {
      // Очищаем таймер переподключения если он был установлен
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }

      // Создаем новое WebSocket соединение
      socket = new WebSocket(WS_URL);

      // Обработчик открытия соединения
      socket.onopen = () => {
        console.log('WebSocket соединение установлено');
        setWsConnected(true);
      };

      // Обработчик сообщений
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Получены данные:', data);

          if (data.sector) {
            console.log('Получены данные о секторе:', data.sector);
            updateSectorDisplay(data.sector);
          } else if (data.status === 'connected') {
            console.log('Получено подтверждение соединения:', data.message);
          }
        } catch (error) {
          console.error('Ошибка при обработке сообщения:', error);
        }
      };

      // Обработчик ошибок
      socket.onerror = (error) => {
        console.error('Ошибка WebSocket:', error);
        setWsConnected(false);
      };

      // Обработчик закрытия соединения
      socket.onclose = () => {
        console.log('WebSocket соединение закрыто, переподключение...');
        setWsConnected(false);

        // Планируем переподключение
        reconnectTimer = setTimeout(() => {
          console.log('Попытка переподключения...');
          connect();
        }, RECONNECT_INTERVAL);
      };
    };

    // Устанавливаем первоначальное соединение
    connect();

    // Очистка при размонтировании компонента
    return () => {
      if (socket) {
        socket.close();
      }
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
    };
  }, [updateSectorDisplay]); // Зависимость от updateSectorDisplay

  return (
    <div className="App">
      <div className="connection-status">
        <span className={wsConnected ? 'connected' : 'disconnected'}>
          • {wsConnected ? 'Подключено' : 'Отключено'}
        </span>
      </div>
      <nav>
        <main>
          <div className="square">
            <RulerTopBottom />
            <Radar showImage={showImage} imageSrc={box} isRed={isRed} isRed2={isRed2} isRed3={isRed3} />
            <RulerRightLeft />
          </div>
        </main>
        <aside>
          <div className="aside-container">
            <h2>Информация</h2>

            <div className="distance-info">
              <p>Тип объекта:</p>
              <div className="info-box">
                <p>{object}</p>
              </div>
            </div>

            <div className="distance-info">
              <p>Расстояние до цели:</p>
              <div className="info-box">
                <p>{way}{edinica}</p>
              </div>
            </div>

            <div className="distance-info">
              <p>Скорость объекта:</p>
              <div className="info-box">
                <p>{way}{edinica}</p>
              </div>
            </div>

            <div className="distance-info">
              <p>Время подлёта:</p>
              <div className="info-box">
                <p>{way}{edinica}</p>
              </div>
            </div>

            <div className="distance-info">
              <p>Сектор обнаружения:</p>
              <div className="info-box sector-info">
                <p>{currentSector}</p>
              </div>
            </div>

          </div>
        </aside>
      </nav>
    </div>
  );
}

export default App;
