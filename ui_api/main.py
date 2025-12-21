from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from ui_api.routers import eligibility, data_management
from ui_api.routers import evaluations 

from prometheus_client import Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

app = FastAPI(
    title="Clinical Trial Eligibility API",
    description="API for clinical trial patient eligibility assessment",
    version="1.0.0"
)

LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["endpoint", "method"],
)

@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    elapsed = time.time() - start_time

    endpoint = request.url.path
    method = request.method
    LATENCY.labels(endpoint=endpoint, method=method).observe(elapsed)

    return response

@app.get("/metrics")
async def metrics():
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    eligibility.router, 
    prefix="/eligibility", 
    tags=["eligibility"]
)
app.include_router(
    data_management.router, 
    prefix="/data", 
    tags=["data-management"]
)
app.include_router(
    evaluations.router,
    prefix="/evaluations",
    tags=["saved-evaluations"],
)

@app.get("/")
async def root():
    return {
        "message": "Clinical Trial Eligibility API",
        "version": "1.0.0",
        "endpoints": {
            "data_management": "/data",
            "eligibility": "/eligibility",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "ui_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )