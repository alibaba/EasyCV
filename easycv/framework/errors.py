# Copyright (c) Alibaba, Inc. and its affiliates.

UNIMPLEMENTED = '0x4818004000000001'
INTERNAL = '0x4818004000000002'
PERMISSION_DENIED = '0x4818004000000003'
NOT_FOUND = '0x4818004000000004'
INVALID_ARGUMENT = '0x4818004000000005'
TIMEOUT = '0x4818004000000006'


class BaseError(Exception):
    """The base error class for exceptions.
  """
    code = None

    def __init__(self, message, details=None, op=None):
        """Creates a new `OpError` indicating that a particular op failed.

      Args:
        message: The message string describing the failure.
        details: The help message that handle the error.
        op: The `ops.Operation` that failed, if known; otherwise None. During
          eager execution, this field is always `None`.
      """
        super(BaseError, self).__init__()
        self._op = op
        self._message = message
        self._details = details

    @property
    def message(self):
        """The error message that describes the error."""
        return self._message

    @property
    def details(self):
        """The help message that handle the error."""
        return self._details

    @property
    def op(self):
        """The operation that failed, if known.
      Returns:
        The `Operation` that failed, or None.
      """
        return self._op

    @property
    def error_code(self):
        """The integer error code that describes the error."""
        return self.code

    def __str__(self):
        return self.message


class UnimplementedError(BaseError):
    """Raised when an operation has not been implemented."""
    code = UNIMPLEMENTED


class InternalError(BaseError):
    """Raised when the system experiences an internal error."""
    code = INTERNAL


class PermissionDeniedError(BaseError):
    """Raised when the caller does not have permission to run an operation."""
    code = PERMISSION_DENIED


class NotFoundError(BaseError):
    """Raised when a requested entity was not found."""
    code = NOT_FOUND


class InvalidArgumentError(BaseError):
    """Raised when an operation receives an invalid argument."""
    code = INVALID_ARGUMENT


class TimeoutError(BaseError):
    """Raised when an operation takes longer than a predetermined time."""
    code = TIMEOUT
