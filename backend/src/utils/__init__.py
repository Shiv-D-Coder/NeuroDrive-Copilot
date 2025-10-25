"""
Utility functions - Optional helpers for candidates

Candidates can add helper functions here as needed for:
- Text processing
- Chunking strategies
- Metadata handling
- Error handling
etc.
"""

# Example utility functions candidates might implement:

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max length at word boundary"""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + '...'


def extract_file_extension(filename: str) -> str:
    """Extract file extension from filename"""
    parts = filename.split('.')
    return parts[-1].lower() if len(parts) > 1 else ''


def sanitize_text(text: str) -> str:
    """Clean and sanitize text content"""
    # Remove excess whitespace
    text = ' '.join(text.split())
    return text.strip()
