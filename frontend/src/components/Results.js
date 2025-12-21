import React, { useState } from 'react';
import './Results.css';
import PatientNoteHighlight from './PatientNoteHighlight';

function Results({ data }) {
  const [expandedSections, setExpandedSections] = useState({});
  const [saveStatus, setSaveStatus] = useState(null);
  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const llmEvaluation = data?.evaluation?.llm || {};
  const verdict = llmEvaluation.verdict || 'unknown';
  const score = llmEvaluation.score;
  const matches = llmEvaluation.matches || [];
  const nonMatches = llmEvaluation.non_matches || [];
  const unknowns = llmEvaluation.unknowns || [];
  const commentary = llmEvaluation.commentary || '';

  const getVerdictClass = (v) => {
    if (v === 'included') return 'eligible';
    if (v === 'excluded') return 'not-eligible';
    return 'unknown';
  };

  const handleSave = async (validity) => {
    try {
      setSaveStatus(`saving-${validity}`);
      const resp = await fetch(`/evaluations/save?validity=${validity}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || 'Ошибка сохранения');
      }
      setSaveStatus(`ok-${validity}`);
      setTimeout(() => setSaveStatus(null), 2000);
    } catch (e) {
      console.error(e);
      setSaveStatus(`error-${validity}`);
      setTimeout(() => setSaveStatus(null), 3000);
    }
  };

  const getVerdictText = (v) => {
    if (v === 'included') return '✓ ПОДХОДИТ';
    if (v === 'excluded') return '✗ НЕ ПОДХОДИТ';
    return '? НЕИЗВЕСТНО';
  };

  const renderCriterionPair = (pair, index) => (
    <div key={index} className="criterion-pair">
      <div className="pair-feature">
        <strong>Признак пациента:</strong> {pair.feature_name || 'N/A'}
        <div className="pair-value">Значение: {JSON.stringify(pair.feature_value)}</div>
        {pair.feature_evidence_spans && pair.feature_evidence_spans.length > 0 && (
          <div className="evidence">
            <em>Из записи:</em> "{pair.feature_evidence_spans[0].text}"
          </div>
        )}
      </div>
      <div className="pair-criterion">
        <strong>Критерий ({pair.criterion_type}):</strong> {pair.criterion_description}
        {pair.criterion_evidence_spans && pair.criterion_evidence_spans.length > 0 && (
          <div className="evidence">
            <em>Из протокола:</em> "{pair.criterion_evidence_spans[0].text}"
          </div>
        )}
      </div>
      {pair.rationale && (
        <div className="pair-rationale">
          <strong>Обоснование:</strong> {pair.rationale}
        </div>
      )}
    </div>
  );

  return (
    <div className="results">
      <h2>📋 Результаты анализа</h2>
      
      <div className="result-item">
        <h3>Пациент: {data.patient_id}</h3>
        <p><strong>Исследование:</strong> {data.trial_id}</p>
        
        <div className={`verdict ${getVerdictClass(verdict)}`}>
          {getVerdictText(verdict)}
        </div>
        
        {score !== null && score !== undefined && (
          <div className="score">Score: {score}/100</div>
        )}

        {commentary && (
          <div className="commentary-section">
            <h4>💬 Комментарий системы:</h4>
            <p>{commentary}</p>
          </div>
        )}
        <div className="save-buttons">
          <button
            type="button"
            className="btn btn-success"
            onClick={() => handleSave('valid')}
          >
            💾 Сохранить как валидный
          </button>
          <button
            type="button"
            className="btn btn-warning"
            onClick={() => handleSave('invalid')}
          >
            ❗ Сохранить как невалидный
          </button>
          {saveStatus && (
            <span className="save-status">
              {saveStatus.startsWith('saving') && 'Сохранение...'}
              {saveStatus.startsWith('ok') && '✔ Сохранено'}
              {saveStatus.startsWith('error') && '✗ Ошибка сохранения'}
            </span>
          )}
        </div>
        {/* Matched Criteria */}
        {matches.length > 0 && (
          <div className="criteria-section">
            <button 
              className="section-toggle"
              onClick={() => toggleSection('matches')}
            >
              <span className="toggle-icon">
                {expandedSections.matches ? '▼' : '▶'}
              </span>
              <strong>✓ Соответствия ({matches.length})</strong>
            </button>
            {expandedSections.matches && (
              <div className="criteria-list">
                {matches.map((pair, idx) => renderCriterionPair(pair, idx))}
              </div>
            )}
          </div>
        )}

        {/* Non-Matched Criteria */}
        {nonMatches.length > 0 && (
          <div className="criteria-section">
            <button 
              className="section-toggle"
              onClick={() => toggleSection('nonMatches')}
            >
              <span className="toggle-icon">
                {expandedSections.nonMatches ? '▼' : '▶'}
              </span>
              <strong>✗ Несоответствия ({nonMatches.length})</strong>
            </button>
            {expandedSections.nonMatches && (
              <div className="criteria-list">
                {nonMatches.map((pair, idx) => renderCriterionPair(pair, idx))}
              </div>
            )}
          </div>
        )}

        {/* Unknown/Ambiguous */}
        {unknowns.length > 0 && (
          <div className="criteria-section">
            <button 
              className="section-toggle"
              onClick={() => toggleSection('unknowns')}
            >
              <span className="toggle-icon">
                {expandedSections.unknowns ? '▼' : '▶'}
              </span>
              <strong>? Неопределенные ({unknowns.length})</strong>
            </button>
            {expandedSections.unknowns && (
              <div className="criteria-list">
                {unknowns.map((pair, idx) => renderCriterionPair(pair, idx))}
              </div>
            )}
          </div>
        )}

        {/* Raw JSON (collapsible) */}
        <div className="criteria-section">
          <button 
            className="section-toggle"
            onClick={() => toggleSection('raw')}
          >
            <span className="toggle-icon">
              {expandedSections.raw ? '▼' : '▶'}
            </span>
            <strong>🔍 Полный JSON ответ</strong>
          </button>
          {expandedSections.raw && (
            <pre className="raw-json">
              {JSON.stringify(data, null, 2)}
            </pre>
          )}
        </div>

        {/* НОВЫЙ блок с исходным текстом пациента и подсветкой */}
        <PatientNoteHighlight data={data} />
      </div>
    </div>
  );
}

export default Results;