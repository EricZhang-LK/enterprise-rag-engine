from http import HTTPStatus


class EnterpriseRagError(Exception):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ResourceNotFoundError(EnterpriseRagError):
    status_code = HTTPStatus.NOT_FOUND
    error_code = "RESOURCE_NOT_FOUND"


class ValidationFailedError(EnterpriseRagError):
    status_code = HTTPStatus.BAD_REQUEST
    error_code = "VALIDATION_FAILED"
