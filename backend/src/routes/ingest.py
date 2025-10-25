"""
Ingestion Routes - Candidates implement the service logic
These routes are complete, but call IngestionService methods that need implementation
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from ..services.ingestion_service import ingestion_service
from ..types import IngestionStatus

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/start")
async def start_ingestion(background_tasks: BackgroundTasks):
    """
    Start the ingestion process in the background.

    This endpoint is COMPLETE. Candidates implement the
    ingestion_service.start_ingestion() method.
    """
    try:
        # Check if already ingesting
        status = ingestion_service.get_status()
        if status.is_ingesting:
            raise HTTPException(
                status_code=400,
                detail="Ingestion already in progress"
            )

        # Start ingestion in background
        background_tasks.add_task(ingestion_service.start_ingestion)

        return {"message": "Ingestion started"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=IngestionStatus)
async def get_ingestion_status():
    """
    Get the current status of the ingestion process.

    This endpoint is COMPLETE and requires no implementation.
    """
    try:
        return ingestion_service.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
