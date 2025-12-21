import React, { useState } from 'react';
import './TraceView.css';

function buildHighlightedText(text, segments) {
  if (!text) return null;
  if (!segments || !segments.length) {
    return <span>{text}</span>;
  }

  const norm = segments
    .map((s, idx) => ({
      start: Math.max(0, Math.min(s.start, text.length)),
      end: Math.max(0, Math.min(s.end, text.length)),
      className: s.className,
      title: s.title || '',
      inlineStyle: s.inlineStyle || {},
      key: idx,
    }))
    .filter(s => s.end > s.start)
    .sort((a, b) => a.start - b.start || a.end - b.end);

  const merged = [];
  for (const s of norm) {
    if (
      merged.length &&
      s.start <= merged[merged.length - 1].end &&
      s.className === merged[merged.length - 1].className &&
      JSON.stringify(s.inlineStyle) === JSON.stringify(merged[merged.length - 1].inlineStyle)
    ) {
      merged[merged.length - 1].end = Math.max(
        merged[merged.length - 1].end,
        s.end
      );
    } else {
      merged.push({ ...s });
    }
  }

  const result = [];
  let current = 0;
  merged.forEach((seg, i) => {
    if (seg.start > current) {
      result.push(
        <span key={`plain-${i}-${current}`}>
          {text.slice(current, seg.start)}
        </span>
      );
    }
    result.push(
      <span
        key={`hl-${i}`}
        className={seg.className}
        title={seg.title}
        style={seg.inlineStyle}
      >
        {text.slice(seg.start, seg.end)}
      </span>
    );
    current = seg.end;
  });

  if (current < text.length) {
    result.push(
      <span key={`tail-${current}`}>{text.slice(current)}</span>
    );
  }

  return result;
}

// Палитра для разных критериев
const CRITERION_COLORS = [
  '#FFE0B2',
  '#C8E6C9',
  '#BBDEFB',
  '#F8BBD0',
  '#FFF9C4',
  '#D1C4E9',
  '#B2EBF2',
  '#FFCCBC',
  '#E6EE9C',
  '#FFCDD2',
];

function TraceView({ data }) {
  const [activeTab, setActiveTab] = useState('inclusion');

  if (!data) return null;

  const meta = data.inputs?.trial_criteria?.metadata || {};
  const trialInclusion = meta.trial_inclusion_text || '';
  const trialExclusion = meta.trial_exclusion_text || '';

  if (!trialInclusion && !trialExclusion) {
    return null;
  }

  const llm = data.evaluation?.llm || {};
  const matches = llm.matches || [];
  const nonMatches = llm.non_matches || [];
  const unknowns = llm.unknowns || [];

  const inclusionSegs = [];
  const exclusionSegs = [];

  // словарь: criterion_id -> { color, type, description }
  const criterionStyles = new Map();
  let colorIndex = 0;

  const getCriterionStyle = (criterionId, criterionType, description) => {
    if (!criterionId) {
      criterionId = `no-id-${description?.slice(0, 40) || ''}`;
    }

    if (!criterionStyles.has(criterionId)) {
      const color = CRITERION_COLORS[colorIndex % CRITERION_COLORS.length];
      colorIndex += 1;
      criterionStyles.set(criterionId, {
        color,
        type: criterionType,
        description: description || '',
      });
    }
    return criterionStyles.get(criterionId);
  };

  const pushSegsFromPair = (pair) => {
    const critType = (pair.criterion_type || 'inclusion').toLowerCase();
    const critId = pair.criterion_id;
    const desc = pair.criterion_description || '';

    const style = getCriterionStyle(critId, critType, desc);

    (pair.criterion_evidence_spans || []).forEach((s) => {
      const baseClass = 'hl-criterion';
      const typeClass =
        critType === 'inclusion'
          ? 'criterion-include-border'
          : 'criterion-exclude-border';

      const segObj = {
        start: s.start_char,
        end: s.end_char,
        className: `${baseClass} ${typeClass}`,
        title: desc,
        inlineStyle: { backgroundColor: style.color },
      };

      const docId = s.document_id || '';
      if (docId.includes(':inclusion')) {
        inclusionSegs.push(segObj);
      } else if (docId.includes(':exclusion')) {
        exclusionSegs.push(segObj);
      }
    });
  };

  // Добавляем сегменты из всех bucket-ов
  [...matches, ...nonMatches, ...unknowns].forEach((p) => pushSegsFromPair(p));

  // Подготовим легенду на основе criterionStyles
  const legendItems = Array.from(criterionStyles.entries()).map(
    ([criterionId, info]) => ({
      id: criterionId,
      color: info.color,
      type: info.type,
      description: info.description,
    })
  );

  return (
    <div className="trace-view">
      <div className="trace-tabs">
        {trialInclusion && (
          <button
            className={activeTab === 'inclusion' ? 'active' : ''}
            onClick={() => setActiveTab('inclusion')}
          >
            Критерии включения
          </button>
        )}
        {trialExclusion && (
          <button
            className={activeTab === 'exclusion' ? 'active' : ''}
            onClick={() => setActiveTab('exclusion')}
          >
            Критерии исключения
          </button>
        )}
      </div>

      <div className="trace-content">
        {activeTab === 'inclusion' && trialInclusion && (
          <pre className="trace-text">
            {buildHighlightedText(trialInclusion, inclusionSegs)}
          </pre>
        )}
        {activeTab === 'exclusion' && trialExclusion && (
          <pre className="trace-text">
            {buildHighlightedText(trialExclusion, exclusionSegs)}
          </pre>
        )}
      </div>

      {legendItems.length > 0 && (
        <div className="trace-legend">
          <h4>Сопоставление цветов и критериев</h4>
          <div className="trace-legend-items">
            {legendItems.map((item) => (
              <div key={item.id} className="trace-legend-item">
                <span
                  className="trace-legend-color-box"
                  style={{ backgroundColor: item.color }}
                />
                <span className="trace-legend-type">
                  {item.type === 'inclusion' ? 'Включение' : 'Исключение'}
                </span>
                <span className="trace-legend-desc">
                  {item.description || item.id}
                </span>
              </div>
            ))}
          </div>
          <div className="trace-legend-note">
            Зелёная рамка — критерий включения; красная рамка — критерий
            исключения. Цвет фона различает отдельные критерии.
          </div>
        </div>
      )}
    </div>
  );
}

export default TraceView;