# app/core/errors.py

from fastapi import HTTPException, status

class BaseAppException(Exception):
    """Base exception for all application-specific exceptions."""
    
    def __init__(self, detail: str = None):
        self.detail = detail or "An unexpected error occurred"
        super().__init__(self.detail)

class AuthenticationError(BaseAppException):
    """Exception raised for authentication failures."""
    
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(detail)
        
    def to_http_exception(self):
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=self.detail,
            headers={"WWW-Authenticate": "Bearer"}
        )

class PermissionDeniedError(BaseAppException):
    """Exception raised when user doesn't have permission for an action."""
    
    def __init__(self, detail: str = "Permission denied"):
        super().__init__(detail)
        
    def to_http_exception(self):
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=self.detail
        )

class RateLimitExceededError(BaseAppException):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(detail)
        
    def to_http_exception(self):
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=self.detail
        )

class ResourceNotFoundError(BaseAppException):
    """Exception raised when a requested resource is not found."""
    
    def __init__(self, resource_type: str = "Resource", detail: str = None):
        detail = detail or f"{resource_type} not found"
        super().__init__(detail)
        
    def to_http_exception(self):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=self.detail
        )

class InvalidInputError(BaseAppException):
    """Exception raised for invalid input data."""
    
    def __init__(self, detail: str = "Invalid input"):
        super().__init__(detail)
        
    def to_http_exception(self):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=self.detail
        )

class KnowledgeBaseError(BaseAppException):
    """Exception raised for issues with the knowledge base."""
    
    def __init__(self, detail: str = "Knowledge base error"):
        super().__init__(detail)
        
    def to_http_exception(self):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=self.detail
        )

class ServiceUnavailableError(BaseAppException):
    """Exception raised when a required service is unavailable."""
    
    def __init__(self, service_name: str = "Service", detail: str = None):
        detail = detail or f"{service_name} is currently unavailable"
        super().__init__(detail)
        
    def to_http_exception(self):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=self.detail
        )