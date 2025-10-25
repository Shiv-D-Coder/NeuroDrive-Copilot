"""
FastAPI Application - COMPLETE
Main application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging

from .routes import auth, drive, ingest, chat

# Load environment variables
load_dotenv()

# Reduce noisy telemetry logs from ChromaDB/posthog
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)

# Create FastAPI app
app = FastAPI(
    title="NeuroDrive-Copilot API",
    description="API for searching and chatting with Google Drive documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api")
app.include_router(drive.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")
app.include_router(chat.router, prefix="/api")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "NeuroDrive-Copilot API is running"
    }


@app.get("/api/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "google_oauth_configured": bool(
            os.getenv("GOOGLE_CLIENT_ID") and
            os.getenv("GOOGLE_CLIENT_SECRET")
        )
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
