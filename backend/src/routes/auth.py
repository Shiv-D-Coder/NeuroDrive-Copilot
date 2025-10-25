"""
Authentication Routes 
Handles Google OAuth flow
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from ..services.drive_service import drive_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/google")
async def get_auth_url():
    """
    Get Google OAuth authorization URL.
    Frontend redirects user to this URL to start OAuth flow.
    """
    try:
        auth_url = drive_service.get_auth_url()
        return {"url": auth_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/callback")
async def auth_callback(code: str):
    """
    Handle OAuth callback from Google.
    Exchange authorization code for credentials.
    """
    try:
        credentials = drive_service.handle_callback(code)

        # TODO (Optional): Store credentials in session/database
        # For simplicity, we're keeping them in memory
        

        # Redirect to frontend
        return RedirectResponse(url="http://localhost:3000")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")


@router.get("/status")
async def get_auth_status():
    """
    Check if user is currently authenticated.
    Returns authentication status.
    """
    try:
        is_authenticated = drive_service.is_authenticated()
        return {"isAuthenticated": is_authenticated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
