// /home/kirill/projects_2/folium/Xplore/frontend/src/components/SavedResultsPage.js

import React, { useEffect, useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import './SavedResultsPage.css';
import ResultsModal from './ResultsModal'; // NEW

// Простая "гистограмма" по trial_id: сколько included / excluded
function buildStats(validList) {
  const statsMap = new Map();
  for (const row of validList) {
    const trialId = row.trial_id;
    const verdict = row.llm_verdict;
    if (!trialId) continue;
    if (!statsMap.has(trialId)) {
      statsMap.set(trialId, { trial_id: trialId, included: 0, excluded: 0, unknown: 0 });
    }
    const obj = statsMap.get(trialId);
    if (verdict === 'included') obj.included += 1;
    else if (verdict === 'excluded') obj.excluded += 1;
    else obj.unknown += 1;
  }
  return Array.from(statsMap.values());
}

function SavedResultsPage() {
  const [activeTab, setActiveTab] = useState('valid');
  const [validList, setValidList] = useState([]);
  const [invalidList, setInvalidList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [reEvalResult, setReEvalResult] = useState(null);
  const [reEvalLoading, setReEvalLoading] = useState(false);
  const [error, setError] = useState(null);

  // NEW: для модалки
  const [selectedEvaluation, setSelectedEvaluation] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [validResp, invalidResp] = await Promise.all([
        fetch('/evaluations/list?validity=valid'),
        fetch('/evaluations/list?validity=invalid'),
      ]);
      if (!validResp.ok || !invalidResp.ok) {
        throw new Error('Ошибка загрузки сохранённых результатов');
      }
      const validData = await validResp.json();
      const invalidData = await invalidResp.json();
      setValidList(validData || []);
      setInvalidList(invalidData || []);
    } catch (e) {
      console.error(e);
      setError(e.message || 'Ошибка загрузки');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const stats = useMemo(() => buildStats(validList), [validList]);

  // NEW: агрегированные метрики
  const dashboard = useMemo(() => {
    const all = [...validList, ...invalidList];
    const total = all.length;
    const included = all.filter(r => r.llm_verdict === 'included').length;
    const excluded = all.filter(r => r.llm_verdict === 'excluded').length;
    const unknown = all.filter(r => !r.llm_verdict || r.llm_verdict === 'unknown').length;

    const validCount = validList.length;
    const invalidCount = invalidList.length;

    const pct = (n) => (total ? Math.round((n / total) * 100) : 0);

    return {
      total,
      included,
      excluded,
      unknown,
      validCount,
      invalidCount,
      pctIncluded: pct(included),
      pctExcluded: pct(excluded),
      pctUnknown: pct(unknown),
    };
  }, [validList, invalidList]);

  const handleReEvaluate = async (id) => {
    try {
      setReEvalLoading(true);
      setReEvalResult(null);
      const resp = await fetch(`/evaluations/re-evaluate/${id}`, {
        method: 'POST',
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || 'Ошибка перезапуска оценки');
      }
      const data = await resp.json();
      setReEvalResult(data);
    } catch (e) {
      console.error(e);
      setError(e.message || 'Ошибка перезапуска оценки');
    } finally {
      setReEvalLoading(false);
    }
  };

  const handleExport = (validity, fmt) => {
    const url =
      fmt === 'json'
        ? `/evaluations/export/json?validity=${validity}`
        : `/evaluations/export/csv?validity=${validity}`;
    window.open(url, '_blank');
  };

  const handleViewReport = async (row) => {
    try {
        setError(null);
        const resp = await fetch(`/evaluations/detail/${row.id}`);
        if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || 'Не удалось загрузить полный отчёт');
        }
        const data = await resp.json();
        setSelectedEvaluation(data.evaluation_json || data);
        setIsModalOpen(true);
    } catch (e) {
        console.error(e);
        setError(e.message || 'Ошибка загрузки отчёта');
    }
    };

  const renderTable = (rows, isInvalidTab) => (
    <table className="saved-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Patient ID</th>
          <th>Trial ID</th>
          <th>Verdict</th>
          <th>Score</th>
          <th>Created At</th>
          <th>Отчёт</th>
          {isInvalidTab && <th>Действия</th>}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id}>
            <td>{r.id}</td>
            <td>{r.patient_id}</td>
            <td>{r.trial_id}</td>
            <td>{r.llm_verdict || '—'}</td>
            <td>
              {r.llm_score !== null && r.llm_score !== undefined
                ? r.llm_score
                : '—'}
            </td>
            <td>{r.created_at ? String(r.created_at) : '—'}</td>
            <td>
              <button
                className="btn btn-small btn-outline"
                type="button"
                onClick={() => handleViewReport(r)}
              >
                👁 Смотреть отчёт
              </button>
            </td>
            {isInvalidTab && (
              <td>
                <button
                  className="btn btn-small"
                  onClick={() => handleReEvaluate(r.id)}
                  disabled={reEvalLoading}
                >
                  🔁 Перезапустить
                </button>
              </td>
            )}
          </tr>
        ))}
      </tbody>
    </table>
  );

  return (
    <div className="saved-page-container">
      <header className="saved-header">
        <div>
          <h1>📂 Сохранённые результаты</h1>
          <p className="saved-subtitle">
            Архив оценок соответствия пациентов критериям клинических исследований
          </p>
        </div>
        <Link to="/" className="btn btn-secondary">
          ← Назад к оценке
        </Link>
      </header>

      {/* NEW: дашборд */}
      <section className="saved-dashboard">
        <div className="dashboard-card">
          <div className="dashboard-title">Всего сохранённых</div>
          <div className="dashboard-value">{dashboard.total}</div>
        </div>
        <div className="dashboard-card">
          <div className="dashboard-title">Валидные / Невалидные</div>
          <div className="dashboard-value">
            {dashboard.validCount} / {dashboard.invalidCount}
          </div>
          <div className="dashboard-bar">
            <div
              className="dashboard-bar-valid"
              style={{
                width:
                  dashboard.validCount + dashboard.invalidCount === 0
                    ? '0%'
                    : `${
                        (dashboard.validCount /
                          (dashboard.validCount + dashboard.invalidCount)) *
                        100
                      }%`,
              }}
            />
          </div>
        </div>
        <div className="dashboard-card">
          <div className="dashboard-title">Вердикты LLM</div>
          <div className="dashboard-pills">
            <span className="pill pill-included">
              ✓ {dashboard.included} ({dashboard.pctIncluded}%)
            </span>
            <span className="pill pill-excluded">
              ✗ {dashboard.excluded} ({dashboard.pctExcluded}%)
            </span>
            <span className="pill pill-unknown">
              ? {dashboard.unknown} ({dashboard.pctUnknown}%)
            </span>
          </div>
        </div>
      </section>

      {error && (
        <div className="error-message">
          <strong>Ошибка:</strong> {error}
        </div>
      )}

      {loading ? (
        <div>Загрузка...</div>
      ) : (
        <>
          <div className="saved-tabs">
            <button
              className={activeTab === 'valid' ? 'active' : ''}
              onClick={() => setActiveTab('valid')}
            >
              ✅ Валидные ({validList.length})
            </button>
            <button
              className={activeTab === 'invalid' ? 'active' : ''}
              onClick={() => setActiveTab('invalid')}
            >
              ⚠ Невалидные ({invalidList.length})
            </button>
          </div>

          <div className="saved-export-buttons">
            {activeTab === 'valid' ? (
              <>
                <button
                  className="btn btn-small"
                  onClick={() => handleExport('valid', 'json')}
                >
                  ⬇ Экспорт JSON
                </button>
                <button
                  className="btn btn-small"
                  onClick={() => handleExport('valid', 'csv')}
                >
                  ⬇ Экспорт CSV
                </button>
              </>
            ) : (
              <>
                <button
                  className="btn btn-small"
                  onClick={() => handleExport('invalid', 'json')}
                >
                  ⬇ Экспорт JSON
                </button>
                <button
                  className="btn btn-small"
                  onClick={() => handleExport('invalid', 'csv')}
                >
                  ⬇ Экспорт CSV
                </button>
              </>
            )}
          </div>

          {activeTab === 'valid' && (
            <div className="saved-content">
              <h2>Валидные результаты</h2>
              {renderTable(validList, false)}

              <div className="stats-block">
                <h3>Статистика по исследованиям (trial_id)</h3>
                {stats.length === 0 ? (
                  <p>Нет данных.</p>
                ) : (
                  <table className="stats-table">
                    <thead>
                      <tr>
                        <th>Trial ID</th>
                        <th>Включено</th>
                        <th>Исключено</th>
                        <th>Неопределено</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stats.map((s) => (
                        <tr key={s.trial_id}>
                          <td>{s.trial_id}</td>
                          <td>{s.included}</td>
                          <td>{s.excluded}</td>
                          <td>{s.unknown}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          )}

          {activeTab === 'invalid' && (
            <div className="saved-content">
              <h2>Невалидные результаты</h2>
              {renderTable(invalidList, true)}

              {reEvalLoading && <div>Перезапуск оценки...</div>}

              {reEvalResult && (
                <div className="reeval-result">
                  <h3>Результат повторной оценки (фрагмент)</h3>
                  <pre>
                    {JSON.stringify(
                      reEvalResult.evaluation?.llm || reEvalResult,
                      null,
                      2
                    )}
                  </pre>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Модальное окно с полным отчётом */}
      <ResultsModal
        open={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        data={selectedEvaluation}
      />
    </div>
  );
}

export default SavedResultsPage;