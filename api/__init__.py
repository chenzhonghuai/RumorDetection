"""
API模块
"""

from .app import create_app
from .detector import RumorDetector

__all__ = ["create_app", "RumorDetector"]
