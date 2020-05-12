"""Custom fdet erros module"""

class DetectorError(Exception):
    """Base class for detector exceptions"""

class DetectorIOError(DetectorError):
    """Base class for IO exceptions"""

class DetectorValueError(DetectorError):
    """Base class for argument value exceptions"""

class DetectorModelError(DetectorError):
    """Base class for model and weights exceptions"""

class DetectorCudaError(DetectorError):
    """Raised when an CUDA related error occurs"""

class DetectorInputError(DetectorError):
    """Raised when image or bach input error occurs"""
