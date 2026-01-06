class ModelServiceError(Exception):
    """
    Base exception for all model service related errors
    (LLM calls, parsing issues, invalid responses, etc.).
    """

    def __init__(self, message: str, *, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()} (caused by {repr(self.cause)})"
        return super().__str__()
