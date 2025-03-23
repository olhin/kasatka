import React from 'react';
import Gradus from './LineInCircle/Gradus/Gradus';
import classes from './Radar.module.css';
import LineInCircle from './LineInCircle/LineInCircle';

const Radar = ({ showImage, imageSrc, isRed, isRed2, isRed3 }) => {
    const hasActiveSector = isRed || isRed2 || isRed3;

    return (
        <div className={classes.radarContainer}>
            <LineInCircle />
            <div className={`${classes.radar} ${hasActiveSector ? classes.hasActiveSector : ''}`}>
                {/* Секторы для кнопки "Снизу" */}
                <div className={`${classes.sector} ${classes['sector-4']} ${isRed3 ? classes.active : ''}`} style={{ transform: 'rotate(90deg)' }}></div>
                {/* <div className={`${classes.sector} ${classes['sector-4']} ${isRed3 ? classes.active : ''}`} style={{ transform: 'rotate(180deg)' }}></div> */}

                {/* Секторы для кнопки "Сверху справа" */}
                <div className={`${classes.sector} ${classes['sector-4']} ${isRed2 ? classes.active : ''}`} style={{ transform: 'rotate(315deg)' }}></div>
                <div className={`${classes.sector} ${classes['sector-4']} ${isRed2 ? classes.active : ''}`} style={{ transform: 'rotate(0deg)' }}></div>

                {/* Секторы для кнопки "Сверху слева" */}
                <div className={`${classes.sector} ${classes['sector-4']} ${isRed ? classes.active : ''}`} style={{ transform: 'rotate(180deg)' }}></div>
                <div className={`${classes.sector} ${classes['sector-4']} ${isRed ? classes.active : ''}`} style={{ transform: 'rotate(225deg)' }}></div>

                {/* Внутренние круги */}
                <div className={classes.innerCircle}></div>
                <div className={classes.innerCircle2}></div>
                <div className={classes.innerCircle3}></div>

                {/* Добавление вращающегося треугольника */}
                <div className={classes.triangle}></div>
            </div>
            <Gradus />

            {/* Тут должна отображаться картинка при нажатии на кнопку */}
            {showImage && (
                <img
                    src={imageSrc}
                    alt="Пример"
                    className={classes.imgDron}
                />
            )}

            <div className={classes.degreeMarkings}>
                {Array.from({ length: 36 }).map((_, index) => (
                    <div
                        key={index}
                        className={`${classes.degreeMarking} ${classes[`rotate-${index * 10}`]}`}
                    ></div>
                ))}
            </div>
        </div>
    );
};

export default Radar;
