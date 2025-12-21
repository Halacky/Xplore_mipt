import React, { useState } from 'react';
import './App.css';
import ManualInput from './components/ManualInput';
import Results from './components/Results';
import Loader from './components/Loader';
import { Link } from 'react-router-dom';

function App() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleEvaluate = async (formData) => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      // Сначала сохраняем пациента
      const patientResponse = await fetch('/data/patient', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          patient_id: formData.patientId,
          note: formData.patientNote,
          metadata: {}
        }),
      });

      if (!patientResponse.ok) {
        throw new Error('Failed to save patient data');
      }

      // Затем сохраняем протокол
      const trialResponse = await fetch('/data/trial-protocol', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          trial_id: formData.trialId,
          trial_title: formData.trialTitle,
          trial_inclusion: formData.trialInclusion,
          trial_exclusion: formData.trialExclusion,
          metadata: {}
        }),
      });

      if (!trialResponse.ok) {
        throw new Error('Failed to save trial protocol');
      }

      // Запускаем оценку
      const evaluationResponse = await fetch(
        `/eligibility/evaluate_by_ids_v2?patient_id=${formData.patientId}&trial_id=${formData.trialId}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      if (!evaluationResponse.ok) {
        const errorData = await evaluationResponse.json();
        throw new Error(errorData.detail || 'Evaluation failed');
      }

      const data = await evaluationResponse.json();
      setResults(data);
    } catch (err) {
      console.error('Error:', err);
      setError(err.message || 'Произошла ошибка при обработке запроса');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <h1>🏥 Система отбора пациентов для клинических исследований</h1>
        <p>Автоматизированный анализ соответствия пациентов критериям исследований</p>
        <div style={{ marginTop: '8px' }}>
          <Link to="/saved" className="btn btn-secondary">
            📂 Сохранённые результаты
          </Link>
        </div>
      </div>

      <div className="content">
        <ManualInput onSubmit={handleEvaluate} disabled={loading} />
        
        {error && (
          <div className="error-message">
            <strong>Ошибка:</strong> {error}
          </div>
        )}
        
        {loading && <Loader />}
        
        {results && !loading && <Results data={results} />}
      </div>
    </div>
  );
}

export default App;