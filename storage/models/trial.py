# storage/models/trial.py
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TrialProtocol:
    """
    Represents a clinical trial protocol stored in DB.
    High-level wrapper, not necessarily the full text.
    """
    trial_id: str
    title: str
    raw_protocol_document_id: str
    metadata: Dict[str, Any]