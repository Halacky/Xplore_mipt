// src/components/PatientNoteHighlight.js

import React, { useMemo } from 'react';
import './PatientNoteHighlight.css';

function buildHighlightedText(text, segments) {
  if (!text) return null;
  if (!segments || !segments.length) return <span>{text}</span>;

  const norm = segments
    .map((s, idx) => ({
      start: Math.max(0, Math.min(s.start, text.length)),
      end: Math.max(0, Math.min(s.end, text.length)),
      className: s.className || '',
      inlineStyle: s.inlineStyle || {},
      title: s.title || '',
      key: idx,
    }))
    .filter(s => s.end > s.start)
    .sort((a, b) => a.start - b.start || a.end - b.end);

  const merged = [];
  for (const s of norm) {
    const last = merged[merged.length - 1];
    if (
      last &&
      s.start <= last.end &&
      s.className === last.className &&
      JSON.stringify(s.inlineStyle) === JSON.stringify(last.inlineStyle)
    ) {
      last.end = Math.max(last.end, s.end);
    } else {
      merged.push({ ...s });
    }
  }

  const out = [];
  let cur = 0;

  merged.forEach((seg, i) => {
    if (seg.start > cur) {
      out.push(
        <span key={`plain-${i}-${cur}`}>
          {text.slice(cur, seg.start)}
        </span>
      );
    }
    out.push(
      <span
        key={`hl-${i}`}
        className={seg.className}
        style={seg.inlineStyle}
        title={seg.title}
      >
        {text.slice(seg.start, seg.end)}
      </span>
    );
    cur = seg.end;
  });

  if (cur < text.length) {
    out.push(<span key={`tail-${cur}`}>{text.slice(cur)}</span>);
  }

  return out;
}

// Палитра фонов для разных признаков
const FEATURE_COLORS = [
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
  '#DCEDC8',
  '#F0F4C3',
];

function PatientNoteHighlight({ data }) {
  // Move useMemo to the top, before any conditional returns
  const { segments, legendItems } = useMemo(() => {
    // Return empty values if no data
    if (!data) {
      return { segments: [], legendItems: [] };
    }

    const noteText = data.inputs?.patient_note_text || '';
    
    const llm = data.evaluation?.llm || {};
    const matches = llm.matches || [];
    const nonMatches = llm.non_matches || [];
    const unknowns = llm.unknowns || [];

    const segs = [];
    const legendMap = new Map(); // feature_id -> { color, name, kind }
    let colorIndex = 0;

    const getFeatureStyle = (featureId, featureName, kind) => {
      const key = featureId || `no-id-${featureName || ''}-${kind}`;
      if (!legendMap.has(key)) {
        const color = FEATURE_COLORS[colorIndex % FEATURE_COLORS.length];
        colorIndex += 1;
        legendMap.set(key, {
          id: key,
          feature_id: featureId,
          name: featureName || '(без имени)',
          kind, // match|non_match|unknown
          color,
        });
      }
      return legendMap.get(key);
    };

    // >>> НАСТРОЙКИ РАСШИРЕНИЯ МЕЛКИХ СПАНОВ <<<
    const MIN_LEN = 5;   // если спан короче этого
    const TARGET_LEN = 10; // хотим примерно столько длины
    const HALF_EXTRA = Math.floor((TARGET_LEN - MIN_LEN) / 2); // примерно 2–3 символа в каждую сторону

    const pushFromBucket = (bucket, kind) => {
      bucket.forEach(pair => {
        const fId = pair.feature_id || '';
        const fName = pair.feature_name || '';
        const styleInfo = getFeatureStyle(fId, fName, kind);

        (pair.feature_evidence_spans || []).forEach(s => {
          if (
            !Number.isInteger(s.start_char) ||
            !Number.isInteger(s.end_char)
          ) {
            return;
          }

          let start = s.start_char;
          let end = s.end_char;

          // реальная длина
          const spanLen = end - start;
          if (spanLen > 0 && spanLen < MIN_LEN) {
            // расширяем спан до TARGET_LEN (по возможности)
            const need = TARGET_LEN - spanLen;     // сколько символов добавить в сумме
            const leftExtra = Math.floor(need / 2);
            const rightExtra = need - leftExtra;

            // сначала «сырое» расширение
            let newStart = start - leftExtra;
            let newEnd = end + rightExtra;

            // ограничиваем диапазоном текста
            newStart = Math.max(0, newStart);
            newEnd = Math.min(noteText.length, newEnd);

            // если по краям не получилось набрать TARGET_LEN, оставляем как есть, но уже расширенным
            start = newStart;
            end = newEnd;
          }

          // финальная защита
          if (end <= start) return;

          const baseClass = 'pn-hl-feature';
          let kindClass = '';
          if (kind === 'match') {
            kindClass = 'pn-border-include';
          } else if (kind === 'non_match') {
            kindClass = 'pn-border-exclude';
          } else {
            kindClass = 'pn-border-unknown';
          }

          segs.push({
            start,
            end,
            className: `${baseClass} ${kindClass}`,
            inlineStyle: {
              backgroundColor: styleInfo.color,
            },
            title: `${fName || 'Признак'} (${kind})`,
          });
        });
      });
    };

    pushFromBucket(matches, 'match');
    pushFromBucket(nonMatches, 'non_match');
    pushFromBucket(unknowns, 'unknown');

    const legendItems = Array.from(legendMap.values());
    return { segments: segs, legendItems };
  }, [data]); // Add data as dependency

  // Now we can do our conditional returns after hooks
  if (!data) return null;

  const noteText = data.inputs?.patient_note_text;
  if (!noteText) return null;

  return (
    <div className="patient-note-block">
      <h3>Исходный текст пациента (с подсветкой признаков)</h3>

      <div className="patient-note-legend">
        <div className="patient-note-legend-note">
          <span className="legend-border legend-border-include" /> — признак,
          поддерживающий включение
          <br />
          <span className="legend-border legend-border-exclude" /> — признак,
          приводящий к исключению
          <br />
          <span className="legend-border legend-border-unknown" /> — признак с
          неопределённым влиянием
        </div>
      </div>

      <pre className="patient-note-text">
        {buildHighlightedText(noteText, segments)}
      </pre>

      {legendItems.length > 0 && (
        <div className="patient-note-legend-features">
          <h4>Сопоставление цветов и признаков</h4>
          <div className="patient-note-legend-list">
            {legendItems.map(item => (
              <div key={item.id} className="patient-note-legend-item">
                <span
                  className="patient-note-legend-color-box"
                  style={{ backgroundColor: item.color }}
                />

                <span className="patient-note-legend-name">
                  {item.name || item.feature_id || '(без имени)'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default PatientNoteHighlight;