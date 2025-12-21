// /home/kirill/projects_2/folium/Xplore/frontend/src/components/ResultsModal.js

import React from 'react';
import './ResultsModal.css';
import Results from './Results';
import TraceView from './TraceView';

function ResultsModal({ open, onClose, data }) {
  if (!open || !data) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-content"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>
            Отчёт по пациенту {data.patient_id} и исследованию {data.trial_id}
          </h2>
          <button className="modal-close-btn" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="modal-body">
          {/* Основной результат */}
          <Results data={data} />

          {/* TraceView для критериев (если есть) */}
          <TraceView data={data} />
        </div>
      </div>
    </div>
  );
}

export default ResultsModal;