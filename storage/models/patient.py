# storage/models/patient.py
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PatientRecord:
    """
    Represents a patient record stored in DB.
    """
    patient_id: str
    note_text: str
    metadata: Dict[str, Any]
    