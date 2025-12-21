from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import pandas as pd
import os
from datetime import datetime
import shutil
from pathlib import Path

from storage.db.session import DBSession
from ui_api.schemas import (
    PatientInput,
    TrialProtocolInput,
    PatientTrialInput,
    PatientResponse,
    TrialProtocolResponse,
    SaveResponse,
    FileUploadResponse
)

router = APIRouter()

# Configuration
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
PATIENTS_CSV_DIR = UPLOAD_DIR / "patients_csv"
PROTOCOLS_PDF_DIR = UPLOAD_DIR / "protocols_pdf"
PATIENTS_CSV_DIR.mkdir(exist_ok=True)
PROTOCOLS_PDF_DIR.mkdir(exist_ok=True)


def get_db_session() -> DBSession:
    """Dependency to get database session"""
    db_params = {
        "url": os.getenv("DATABASE_URL", "sqlite:///./clinical_trials.db"),
        "echo": False
    }
    db = DBSession(db_params)
    db.init_db()
    return db


@router.post("/patient", response_model=SaveResponse)
async def save_patient(
    patient: PatientInput,
    db: DBSession = Depends(get_db_session)
):
    """
    Save or update patient data
    
    Args:
        patient: Patient data including patient_id and clinical note
    
    Returns:
        SaveResponse with status and saved data
    """
    try:
        print("=" * 80)
        print("PATIENT DATA RECEIVED:")
        print(f"Patient ID: {patient.patient_id}")
        print(f"Note length: {len(patient.note)} characters")
        print(f"Note preview: {patient.note[:200]}...")
        print(f"Metadata: {patient.metadata}")
        print("=" * 80)
        
        # Save to database
        patient_data = {
            "patient_id": patient.patient_id,
            "note": patient.note,
            "metadata": patient.metadata
        }
        db.save_patient(patient_data)
        
        # Retrieve saved patient (returns dict)
        saved_patient = db.get_patient(patient.patient_id)
        
        if not saved_patient:
            raise HTTPException(status_code=500, detail="Failed to retrieve saved patient")
        
        return SaveResponse(
            status="success",
            message=f"Patient {patient.patient_id} saved successfully",
            data={
                "patient_id": saved_patient["patient_id"],
                "note_length": len(saved_patient["note"]),
                "created_at": str(saved_patient["created_at"]) if saved_patient.get("created_at") else None
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving patient: {str(e)}")


@router.post("/trial-protocol", response_model=SaveResponse)
async def save_trial_protocol(
    trial: TrialProtocolInput,
    db: DBSession = Depends(get_db_session)
):
    """
    Save or update trial protocol data
    
    Args:
        trial: Trial protocol data including criteria
    
    Returns:
        SaveResponse with status and saved data
    """
    try:
        print("=" * 80)
        print("TRIAL PROTOCOL DATA RECEIVED:")
        print(f"Trial ID: {trial.trial_id}")
        print(f"Trial Title: {trial.trial_title}")
        if trial.trial_inclusion:
            print(f"Inclusion Criteria length: {len(trial.trial_inclusion)} characters")
            print(f"Inclusion preview: {trial.trial_inclusion[:200]}...")
        if trial.trial_exclusion:
            print(f"Exclusion Criteria length: {len(trial.trial_exclusion)} characters")
            print(f"Exclusion preview: {trial.trial_exclusion[:200]}...")
        print(f"Metadata: {trial.metadata}")
        print("=" * 80)
        
        # Save to database
        trial_data = {
            "trial_id": trial.trial_id,
            "trial_title": trial.trial_title,
            "trial_inclusion": trial.trial_inclusion,
            "trial_exclusion": trial.trial_exclusion,
            "metadata": trial.metadata
        }
        db.save_trial_protocol(trial_data)
        
        # Retrieve saved trial (returns dict)
        saved_trial = db.get_trial(trial.trial_id)
        
        if not saved_trial:
            raise HTTPException(status_code=500, detail="Failed to retrieve saved trial")
        
        return SaveResponse(
            status="success",
            message=f"Trial protocol {trial.trial_id} saved successfully",
            data={
                "trial_id": saved_trial["trial_id"],
                "trial_title": saved_trial["trial_title"],
                "has_inclusion": saved_trial["trial_inclusion"] is not None,
                "has_exclusion": saved_trial["trial_exclusion"] is not None,
                "created_at": str(saved_trial["created_at"]) if saved_trial.get("created_at") else None
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving trial protocol: {str(e)}")


@router.post("/patient-trial-combined", response_model=SaveResponse)
async def save_patient_trial_combined(
    data: PatientTrialInput,
    db: DBSession = Depends(get_db_session)
):
    """
    Save both patient and trial protocol data in one request
    
    Args:
        data: Combined patient and trial data
    
    Returns:
        SaveResponse with status and saved data
    """
    try:
        print("=" * 80)
        print("COMBINED PATIENT-TRIAL DATA RECEIVED:")
        print(f"Patient ID: {data.patient_id}")
        print(f"Note length: {len(data.note)} characters")
        print(f"Trial ID: {data.trial_id}")
        print(f"Trial Title: {data.trial_title}")
        if data.trial_inclusion:
            print(f"Inclusion Criteria length: {len(data.trial_inclusion)} characters")
        if data.trial_exclusion:
            print(f"Exclusion Criteria length: {len(data.trial_exclusion)} characters")
        print("=" * 80)
        
        # Save patient
        patient_data = {
            "patient_id": data.patient_id,
            "note": data.note,
            "metadata": {}
        }
        db.save_patient(patient_data)
        
        # Save trial protocol
        trial_data = {
            "trial_id": data.trial_id,
            "trial_title": data.trial_title,
            "trial_inclusion": data.trial_inclusion,
            "trial_exclusion": data.trial_exclusion,
            "metadata": {}
        }
        db.save_trial_protocol(trial_data)
        
        return SaveResponse(
            status="success",
            message=f"Patient {data.patient_id} and Trial {data.trial_id} saved successfully",
            data={
                "patient_id": data.patient_id,
                "trial_id": data.trial_id
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving combined data: {str(e)}")


@router.post("/upload-files", response_model=FileUploadResponse)
async def upload_files(
    patients_csv: UploadFile = File(..., description="CSV file with patient data"),
    protocol_pdf: Optional[UploadFile] = File(None, description="PDF file with trial protocol"),
    trial_id: str = Form(..., description="Trial ID for the protocol"),
    db: DBSession = Depends(get_db_session)
):
    """
    Upload and process CSV file with patients and optional PDF protocol
    
    Args:
        patients_csv: CSV file with columns: patient_id, note, trial_id, etc.
        protocol_pdf: Optional PDF file with trial protocol
        trial_id: Trial ID to associate with the protocol PDF
    
    Returns:
        FileUploadResponse with status and file locations
    """
    try:
        files_saved = {}
        records_processed = 0
        
        if patients_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"patients_{timestamp}_{patients_csv.filename}"
            csv_path = PATIENTS_CSV_DIR / csv_filename
            
            with open(csv_path, "wb") as buffer:
                shutil.copyfileobj(patients_csv.file, buffer)
            
            files_saved["patients_csv"] = str(csv_path)
            
            print("=" * 80)
            print(f"CSV FILE UPLOADED: {csv_path}")
            
            # Parse and save patients from CSV
            try:
                df = pd.read_csv(csv_path)
                print(f"CSV contains {len(df)} rows")
                print(f"Columns: {list(df.columns)}")
                
                for idx, row in df.iterrows():
                    try:
                        # Save patient if patient_id and note exist
                        if "patient_id" in row and "note" in row:
                            patient_data = {
                                "patient_id": str(row["patient_id"]),
                                "note": str(row["note"]),
                                "metadata": {
                                    "source_file": csv_filename,
                                    "row_index": int(idx)
                                }
                            }
                            db.save_patient(patient_data)
                            records_processed += 1
                        
                        # Save trial protocol if exists in row
                        if "trial_id" in row:
                            trial_data = {
                                "trial_id": str(row["trial_id"]),
                                "trial_title": str(row.get("trial_title", "")),
                                "trial_inclusion": str(row.get("trial_inclusion", "")),
                                "trial_exclusion": str(row.get("trial_exclusion", "")),
                                "metadata": {
                                    "source_file": csv_filename
                                }
                            }
                            db.save_trial_protocol(trial_data)
                    except Exception as row_error:
                        print(f"Error processing row {idx}: {str(row_error)}")
                        continue
                
                print(f"Successfully processed {records_processed} patient records")
            except Exception as csv_error:
                print(f"Error parsing CSV: {str(csv_error)}")
        
        # Process PDF file
        if protocol_pdf:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"protocol_{trial_id}_{timestamp}_{protocol_pdf.filename}"
            pdf_path = PROTOCOLS_PDF_DIR / pdf_filename
            
            with open(pdf_path, "wb") as buffer:
                shutil.copyfileobj(protocol_pdf.file, buffer)
            
            files_saved["protocol_pdf"] = str(pdf_path)
            
            print(f"PDF FILE UPLOADED: {pdf_path}")
            
            # Update trial protocol with PDF path
            trial_data = {
                "trial_id": trial_id,
                "protocol_pdf_path": str(pdf_path),
                "metadata": {
                    "pdf_filename": pdf_filename,
                    "uploaded_at": timestamp
                }
            }
            db.save_trial_protocol(trial_data)
        
        print("=" * 80)
        
        return FileUploadResponse(
            status="success",
            message="Files uploaded and processed successfully",
            files_saved=files_saved,
            records_processed=records_processed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")


@router.get("/patient/{patient_id}", response_model=Dict[str, Any])
async def get_patient(
    patient_id: str,
    db: DBSession = Depends(get_db_session)
):
    """Get patient data by ID"""
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    # Convert meta_data to metadata for response
    response = patient.copy()
    response["metadata"] = response.pop("meta_data", {})
    return response


@router.get("/trial-protocol/{trial_id}", response_model=Dict[str, Any])
async def get_trial_protocol(
    trial_id: str,
    db: DBSession = Depends(get_db_session)
):
    """Get trial protocol data by ID"""
    trial = db.get_trial(trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail=f"Trial {trial_id} not found")
    
    # Convert meta_data to metadata for response
    response = trial.copy()
    response["metadata"] = response.pop("meta_data", {})
    return response