"""
Drive Routes 
Handles file and folder listing from Google Drive
"""

from fastapi import APIRouter, HTTPException
from typing import List
from ..services.drive_service import drive_service
from ..types import DriveFile, DriveFolder

router = APIRouter(prefix="/drive", tags=["drive"])


@router.get("/files", response_model=List[DriveFile])
async def list_files():
    """
    List all files in Google Drive.
    Returns formatted list of DriveFile objects.
    """
    try:
        files = drive_service.list_files()

        # Convert to DriveFile objects with paths
        drive_files = []
        for file in files:
            path = drive_service.build_file_path(
                file['id'],
                file['name'],
                file.get('parents')
            )

            drive_file = DriveFile(
                id=file['id'],
                name=file['name'],
                mimeType=file['mimeType'],
                path=path,
                modifiedTime=file['modifiedTime'],
                size=file.get('size'),
                webViewLink=file.get('webViewLink')
            )
            drive_files.append(drive_file)

        return drive_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/folders", response_model=List[DriveFolder])
async def list_folders():
    """
    List all folders in Google Drive.
    Returns formatted list of DriveFolder objects with file counts.
    """
    try:
        files = drive_service.list_files()

        # Filter folders and count files
        folders_map = {}
        for file in files:
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                folders_map[file['id']] = {
                    'id': file['id'],
                    'name': file['name'],
                    'file_count': 0,
                    'parents': file.get('parents')
                }

        # Count files in each folder
        for file in files:
            if file['mimeType'] != 'application/vnd.google-apps.folder':
                parents = file.get('parents', [])
                for parent_id in parents:
                    if parent_id in folders_map:
                        folders_map[parent_id]['file_count'] += 1

        # Convert to DriveFolder objects
        drive_folders = []
        for folder_data in folders_map.values():
            path = drive_service.build_file_path(
                folder_data['id'],
                folder_data['name'],
                folder_data.get('parents')
            )

            drive_folder = DriveFolder(
                id=folder_data['id'],
                name=folder_data['name'],
                path=path,
                fileCount=folder_data['file_count']
            )
            drive_folders.append(drive_folder)

        return drive_folders
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/folders/{folder_id}", response_model=DriveFolder)
async def get_folder(folder_id: str):
    """
    Get details about a specific folder.
    """
    try:
        folder_metadata = drive_service.get_file_metadata(folder_id)

        if folder_metadata['mimeType'] != 'application/vnd.google-apps.folder':
            raise HTTPException(status_code=400, detail="File is not a folder")

        # Count files in folder
        files_in_folder = drive_service.list_files(folder_id=folder_id)
        file_count = len([f for f in files_in_folder if f['mimeType'] != 'application/vnd.google-apps.folder'])

        path = drive_service.build_file_path(
            folder_metadata['id'],
            folder_metadata['name'],
            folder_metadata.get('parents')
        )

        return DriveFolder(
            id=folder_metadata['id'],
            name=folder_metadata['name'],
            path=path,
            fileCount=file_count
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
