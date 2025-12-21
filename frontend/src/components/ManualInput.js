import React, { useState } from 'react';
import './ManualInput.css';

function ManualInput({ onSubmit, disabled }) {
  const [formData, setFormData] = useState({
    patientId: '',
    patientNote: '',
    trialId: '',
    trialTitle: '',
    trialInclusion: '',
    trialExclusion: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  // Тестовые данные для быстрого заполнения
  const fillTestData = () => {
    setFormData({
      patientId: 'emp-038',
      patientNote: '0. A 65-year-old man with chronic HFrEF. 1. NYHA II. 2. LVEF 29%. 3. Sinus rhythm. 4. NT-proBNP 650 pg/mL. 5. His last HF hospitalization was 14 months ago with no admissions since. 6. eGFR 41. 7. SBP 119. 8. On GDMT. 9. No type 1 diabetes and no SGLT2 inhibitor exposure.',
      trialId: 'NCT03057977',
      trialTitle: 'Study to Evaluate Dapagliflozin Effect on Heart Failure',
      trialInclusion: `Male or female patient age >= 18 years at screening. For Japan only: Age >= 20 years at screening
Patients with chronic HF (Chronic Heart Failure) NYHA (New York Heart Association Classification) class II-IV and reduced EF (Ejection Fraction) (LVEF (Left Ventricular Ejection Fraction) <=40%) and elevated NT-proBNP (N-terminal of the prohormone brain natriuretic peptide)

If EF >= 36% to <= 40%: NT-proBNP >= 2500 pg/ml or patients without AF (atrial fibrillation/atrial flutter) and NT-proBNP >= 5000 pg/ml for patients with AF
If EF >= 31% to <= 35%: NT-proBNP >= 1000 pg/ml for patients without AF and NT-proBNP >=2000 pg/ml for patients with AF
If EF<= 30%: NT-proBNP >= 600 pg/ml for patients without AF and NT-proBNP >=1200 pg/ml for patients with AF
EF ≤ 40% and hospitalization for heart failure in the past 12 months: NTproBNP ≥ 600 pg/ml for patients without AF and NT-proBNP >= 1200 pg/ml for patients with AF
Appropriate dose of medical therapy for HF consistent with prevailing local and international CV (Cardiovascular) guidelines, stable for at least 1 week prior to Visit 1
Appropriate use of medical devices such as cardioverter defibrillator (ICD) or a cardiac resynchronization therapy (CRT) consistent with prevailing local or international CV guidelines
Signed and dated written ICF (Informed Consent Form)
Further inclusion criteria apply`,
      trialExclusion: `Myocardial infarction, coronary artery bypass graft surgery, or other major cardiovascular surgery, stroke or TIA (Transient Ischaemic Attack) in past 90 days prior to Visit 1
Heart transplant recipient, or listed for heart transplant
Acute decompensated HF
Systolic blood pressure (SBP) >= 180 mmHg at Visit 2.
Symptomatic hypotension and/or a SBP < 100 mmHg
Indication of liver disease
Impaired renal function, defined as eGFR (Estimated Glomerular Filtration Rate) < 20 mL/min/1.73 m2 (CKD-EPI (Chronic Kidney Disease - Epidemiology Collaboration Equation)) or requiring dialysis
History of ketoacidosis
Current use or prior use of a SGLT (Sodium-glucose co-transporter)-2 inhibitor or combined SGLT-1 and 2 inhibitor
Currently enrolled in another investigational device or drug study
Known allergy or hypersensitivity to empagliflozin or other SGLT-2 inhibitors
Women who are pregnant, nursing, or who plan to become pregnant while in the trial
Further exclusion criteria apply`
    });
  };

  return (
    <form className="manual-form" onSubmit={handleSubmit}>
      <div className="test-data-button-container">
        <button 
          type="button" 
          className="btn-test-data"
          onClick={fillTestData}
        >
          📝 Заполнить тестовыми данными
        </button>
      </div>

      <div className="form-group">
        <label htmlFor="patientId">ID пациента *</label>
        <input
          type="text"
          id="patientId"
          name="patientId"
          value={formData.patientId}
          onChange={handleChange}
          placeholder="Введите ID пациента"
          required
          disabled={disabled}
        />
      </div>

      <div className="form-group">
        <label htmlFor="patientNote">Анамнез пациента *</label>
        <textarea
          id="patientNote"
          name="patientNote"
          value={formData.patientNote}
          onChange={handleChange}
          placeholder="Введите медицинскую историю пациента..."
          required
          disabled={disabled}
        />
      </div>

      <div className="form-group">
        <label htmlFor="trialId">ID исследования *</label>
        <input
          type="text"
          id="trialId"
          name="trialId"
          value={formData.trialId}
          onChange={handleChange}
          placeholder="Введите ID исследования"
          required
          disabled={disabled}
        />
      </div>

      <div className="form-group">
        <label htmlFor="trialTitle">Название исследования *</label>
        <input
          type="text"
          id="trialTitle"
          name="trialTitle"
          value={formData.trialTitle}
          onChange={handleChange}
          placeholder="Введите название исследования"
          required
          disabled={disabled}
        />
      </div>

      <div className="row">
        <div className="form-group">
          <label htmlFor="trialInclusion">Критерии включения *</label>
          <textarea
            id="trialInclusion"
            name="trialInclusion"
            value={formData.trialInclusion}
            onChange={handleChange}
            placeholder="Перечислите критерии включения..."
            required
            disabled={disabled}
          />
        </div>

        <div className="form-group">
          <label htmlFor="trialExclusion">Критерии исключения *</label>
          <textarea
            id="trialExclusion"
            name="trialExclusion"
            value={formData.trialExclusion}
            onChange={handleChange}
            placeholder="Перечислите критерии исключения..."
            required
            disabled={disabled}
          />
        </div>
      </div>

      <button type="submit" className="btn btn-primary" disabled={disabled}>
        🚀 Запустить проверку
      </button>
    </form>
  );
}

export default ManualInput;