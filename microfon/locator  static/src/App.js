import React, { useState } from 'react';
import './App.css';

import Radar from './components/Radar/Radar';
import RulerRightLeft from './components/Ruler/RulerRightLeft';
import RulerTopBottom from './components/Ruler/RulerTopBottom';
import box from './Photo/eebc4b7dc05611ef8ab15ec00c958ca6_1.jpg';

let way = ""
let edinica = ' '
let object = ' '

function App() {
  const [showImage, setShowImage] = useState(false);
  const [isRed, setIsRed] = useState(false);
  const [isRed2, setIsRed2] = useState(false);
  const [isRed3, setIsRed3] = useState(false);

  const handleColorChange = () => {
    setIsRed(true);
    setTimeout(() => {
      setIsRed(false);
    }, 2000);
  };

  const handleColorChange2 = () => {
    setIsRed2(true);
    setTimeout(() => {
      setIsRed2(false);
    }, 2000);
  };

  const handleColorChange3 = () => {
    setIsRed3(true);
    setTimeout(() => {
      setIsRed3(false);
    }, 2000);
  };

  return (
    <div className="App">
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
              <div className="info-box">
                <p>{way}{edinica}</p>
              </div>
            </div>

            <div className="info-buttons">
              <button onClick={handleColorChange3}>Снизу</button>
              <button onClick={handleColorChange2}>Сверху справа</button>
              <button onClick={handleColorChange}>Сверху слева</button>
            </div>
          </div>
        </aside>
      </nav>
    </div>
  );
}

export default App;
