from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class PatientInput(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    note: str = Field(..., description="Patient clinical note")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TrialProtocolInput(BaseModel):
    trial_id: str = Field(..., description="Unique trial identifier")
    trial_title: Optional[str] = Field(None, description="Trial title")
    trial_inclusion: Optional[str] = Field(None, description="Inclusion criteria text")
    trial_exclusion: Optional[str] = Field(None, description="Exclusion criteria text")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PatientTrialInput(BaseModel):
    patient_id: str
    note: str
    trial_id: str
    trial_title: Optional[str] = None
    trial_inclusion: Optional[str] = None
    trial_exclusion: Optional[str] = None


class PatientResponse(BaseModel):
    patient_id: str
    note: str
    meta_data: Dict[str, Any] = Field(alias="metadata")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
        populate_by_name = True


class TrialProtocolResponse(BaseModel):
    trial_id: str
    trial_title: Optional[str]
    trial_inclusion: Optional[str]
    trial_exclusion: Optional[str]
    protocol_pdf_path: Optional[str]
    meta_data: Dict[str, Any] = Field(alias="metadata")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
        populate_by_name = True


class SaveResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


class FileUploadResponse(BaseModel):
    status: str
    message: str
    files_saved: Dict[str, str]
    records_processed: Optional[int] = None