"""
Google Drive Service - Handles authentication and file operations
"""

from typing import List, Optional
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
from dotenv import load_dotenv
load_dotenv()


class DriveService:
    """Service for interacting with Google Drive API"""

    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    def __init__(self):
        self.client_id = os.getenv('GOOGLE_CLIENT_ID')
        self.client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
        self.redirect_uri = os.getenv('GOOGLE_REDIRECT_URI')
        self.credentials: Optional[Credentials] = None

    def get_auth_url(self) -> str:
        """Generate Google OAuth authorization URL"""
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.redirect_uri],
                }
            },
            scopes=self.SCOPES,
            redirect_uri=self.redirect_uri
        )

        auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )

        return auth_url

    def handle_callback(self, code: str) -> Credentials:
        """Handle OAuth callback and exchange code for credentials"""
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.redirect_uri],
                }
            },
            scopes=self.SCOPES,
            redirect_uri=self.redirect_uri
        )

        flow.fetch_token(code=code)
        self.credentials = flow.credentials
        return self.credentials

    def list_files(self, folder_id: Optional[str] = None, page_size: int = 100) -> List[dict]:
        """List all files in Drive or in a specific folder"""
        if not self.credentials:
            raise ValueError("Not authenticated. Call handle_callback first.")

        service = build('drive', 'v3', credentials=self.credentials)

        query = "trashed=false"
        if folder_id:
            query += f" and '{folder_id}' in parents"

        results = []
        page_token = None

        while True:
            response = service.files().list(
                q=query,
                pageSize=page_size,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, size, webViewLink, parents)",
                pageToken=page_token
            ).execute()

            results.extend(response.get('files', []))
            page_token = response.get('nextPageToken')

            if not page_token:
                break

        return results

    def get_file_metadata(self, file_id: str) -> dict:
        """Get metadata for a specific file"""
        if not self.credentials:
            raise ValueError("Not authenticated")

        service = build('drive', 'v3', credentials=self.credentials)
        return service.files().get(
            fileId=file_id,
            fields="id, name, mimeType, modifiedTime, size, webViewLink, parents"
        ).execute()

    def download_file(self, file_id: str) -> bytes:
        """Download file content"""
        if not self.credentials:
            raise ValueError("Not authenticated")

        service = build('drive', 'v3', credentials=self.credentials)
        request = service.files().get_media(fileId=file_id)

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        return fh.getvalue()

    def export_google_doc(self, file_id: str, mime_type: str = 'text/plain') -> bytes:
        """Export Google Docs/Sheets/Slides to a downloadable format"""
        if not self.credentials:
            raise ValueError("Not authenticated")

        service = build('drive', 'v3', credentials=self.credentials)
        request = service.files().export_media(fileId=file_id, mimeType=mime_type)

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        return fh.getvalue()

    def build_file_path(self, file_id: str, file_name: str, parents: Optional[List[str]] = None) -> str:
        """Build the full path of a file by traversing parent folders"""
        if not parents or not self.credentials:
            return f"/{file_name}"

        service = build('drive', 'v3', credentials=self.credentials)
        path_parts = [file_name]

        current_parent = parents[0] if parents else None

        while current_parent:
            try:
                parent_metadata = service.files().get(
                    fileId=current_parent,
                    fields="id, name, parents"
                ).execute()

                path_parts.insert(0, parent_metadata['name'])
                current_parent = parent_metadata.get('parents', [None])[0]
            except:
                break

        return "/" + "/".join(path_parts)

    def is_authenticated(self) -> bool:
        """Check if user is authenticated with valid credentials"""
        return self.credentials is not None and self.credentials.valid


# Global instance
drive_service = DriveService()
