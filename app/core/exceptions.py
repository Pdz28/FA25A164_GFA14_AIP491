"""Custom application exceptions."""
from __future__ import annotations


class AppException(Exception):
    """Base application exception."""
    
    def __init__(self, message: str, status_code: int = 500, details: dict | None = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ServiceNotReadyError(AppException):
    """Raised when service is not initialized."""
    
    def __init__(self, message: str = "Service not ready"):
        super().__init__(message, status_code=503)


class ModelNotLoadedError(AppException):
    """Raised when a required model is not loaded."""
    
    def __init__(self, model_name: str):
        super().__init__(
            f"{model_name} model not loaded",
            status_code=400,
            details={"model": model_name}
        )


class InvalidInputError(AppException):
    """Raised for invalid user input."""
    
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=400, details=details)


class PredictionError(AppException):
    """Raised when prediction fails."""
    
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            f"Prediction failed: {message}",
            status_code=500,
            details=details
        )


class WeightLoadError(AppException):
    """Raised when weight loading fails."""
    
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            f"Weight loading failed: {message}",
            status_code=500,
            details=details
        )
