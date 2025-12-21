import React from 'react';
import './Loader.css';

function Loader() {
  return (
    <div className="loader">
      <div className="spinner"></div>
      <p>Анализ данных... Пожалуйста, подождите</p>
      <p className="loader-hint">
        Это может занять 1-2 минуты из-за работы ensemble моделей
      </p>
    </div>
  );
}

export default Loader;