from sqlalchemy import Column, String, Text, DateTime, JSON, Integer, Index, Boolean
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"
    
    patient_id = Column(String(255), primary_key=True)
    note = Column(Text, nullable=False)
    meta_data = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class TrialProtocol(Base):
    __tablename__ = "trial_protocols"
    
    trial_id = Column(String(255), primary_key=True)
    trial_title = Column(String(500), nullable=True)
    trial_inclusion = Column(Text, nullable=True)
    trial_exclusion = Column(Text, nullable=True)
    protocol_pdf_path = Column(String(500), nullable=True)
    meta_data = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class EligibilityResult(Base):
    __tablename__ = "eligibility_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(255), nullable=False, index=True)
    trial_id = Column(String(255), nullable=False, index=True)
    verdict = Column(String(50), nullable=False)  # included/excluded/unknown
    score = Column(Integer, nullable=True)  # 0-100
    evaluation_json = Column(JSON, nullable=False)  
    expert_eligibility = Column(String(50), nullable=True) 
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_patient_trial', 'patient_id', 'trial_id'),
    )

class SavedEvaluation(Base):
    __tablename__ = "saved_evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(255), nullable=False, index=True)
    trial_id = Column(String(255), nullable=False, index=True)

    # "valid" / "invalid"
    validity = Column(String(50), nullable=False)

    evaluation_json = Column(JSON, nullable=False)

    user_comment = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_saved_eval_patient_trial", "patient_id", "trial_id"),
    )