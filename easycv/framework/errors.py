# Copyright (c) Alibaba, Inc. and its affiliates.

OK = '0x5818004000000000'
RUNTIME = '0x4818004000000001'
UNIMPLEMENTED = '0x4818004000000002'
INVALID_ARGUMENT = '0x4818004000000003'
INVALID_VALUE = '0x4818004000000004'
INVALID_KEY = '0x4818004000000005'
INVALID_TYPE = '0x4818004000000006'
MODULE_NOT_FOUND = '0x4818004000000007'
FILE_NOT_FOUND = '0x4818004000000008'
IO_FAILED = '0x4818004000000009'
PERMISSION_DENIED = '0x481800400000000a'
TIMEOUT = '0x481800400000000b'


class BaseError(Exception):
    """The base error class for exceptions.
  """
    code = None

    def __init__(self, message='', details=None, op=None):
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


class NotImplementedError(BaseError):
    """Raised when an operation has not been implemented."""
    code = UNIMPLEMENTED


class RuntimeError(BaseError):
    """Raised when the system experiences an internal error."""
    code = RUNTIME


class PermissionDeniedError(BaseError):
    """Raised when the caller does not have permission to run an operation."""
    code = PERMISSION_DENIED


class FileNotFoundError(BaseError):
    """Raised when a requested entity was not found."""
    code = FILE_NOT_FOUND


class ModuleNotFoundError(BaseError):
    """Raised when a module could not be located."""
    code = MODULE_NOT_FOUND


class InvalidArgumentError(BaseError):
    """Raised when an operation receives an invalid argument."""
    code = INVALID_ARGUMENT


class TimeoutError(BaseError):
    """Raised when an operation timed out."""
    code = TIMEOUT


class IOError(BaseError):
    """Raised when an operation returns a system-related error, including I/O failures."""
    code = IO_FAILED


class ValueError(BaseError):
    """Raised when an operation receives an invalid value."""
    code = INVALID_VALUE


class KeyError(BaseError):
    """Raised when a mapping (dictionary) key is not found in the set of existing keys."""
    code = INVALID_KEY


class TypeError(BaseError):
    """Raised when an operation or function is applied to an object of inappropriate type."""
    code = INVALID_TYPE