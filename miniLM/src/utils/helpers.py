"""
Helper utilities for MiniChat application.
"""
import re
import uuid
from datetime import datetime


def generate_uuid() -> str:
    """
    Generate a unique identifier.
    
    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def format_timestamp(dt: datetime) -> str:
    """
    Format a datetime object to a human-readable string.
    
    Args:
        dt: Datetime object to format
        
    Returns:
        Formatted timestamp string (e.g., "2024-01-15 14:30:45")
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, adding suffix if truncated.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of result (including suffix)
        suffix: String to append if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncate_at = max_length - len(suffix)
    if truncate_at <= 0:
        return suffix[:max_length]
    
    return text[:truncate_at] + suffix


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    # Remove or replace invalid characters
    # Keep alphanumeric, dots, hyphens, underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip('. ')
    
    # Replace multiple underscores/spaces with single underscore
    sanitized = re.sub(r'[_\s]+', '_', sanitized)
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized
