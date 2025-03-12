"""
This is a pipeline for author disambiguation between Gateway to Research
and OpenAlex author profiles.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1" 