"""Plate detection module. Integration: app.include_router(plate_detection.router)"""
from .router import router, register_hook
from .detector import PlateDetector

__all__ = ["router", "register_hook", "PlateDetector"]
