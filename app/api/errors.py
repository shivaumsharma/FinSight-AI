"""
errors.py

Structured error codes instead of a bare HTTP status + English
sentence -- any client (a future mobile app included) needs to branch
on `code`, not parse a message string to decide what screen to show.

Registered as a FastAPI exception handler in main.py; every endpoint
raises APIError directly rather than returning ad-hoc error dicts, so
the response shape ({"code": ..., "message": ...}) stays consistent
across every failure mode without each endpoint reimplementing it.
"""

from fastapi import status

NO_COMPANY_DETECTED = "NO_COMPANY_DETECTED"
JOB_NOT_FOUND = "JOB_NOT_FOUND"
JOB_NOT_DONE = "JOB_NOT_DONE"
INTERRUPTED = "INTERRUPTED"
TIMEOUT = "TIMEOUT"
UNAUTHORIZED = "UNAUTHORIZED"
INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
EMAIL_ALREADY_REGISTERED = "EMAIL_ALREADY_REGISTERED"
FORBIDDEN = "FORBIDDEN"
RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
TICKER_NOT_FOUND = "TICKER_NOT_FOUND"


class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def no_company_detected(question: str) -> APIError:
    return APIError(
        NO_COMPANY_DETECTED,
        "No publicly listed company was detected in this question. "
        "Please name a company or ticker.",
        status_code=status.HTTP_400_BAD_REQUEST,
    )


def job_not_found(job_id: str) -> APIError:
    return APIError(
        JOB_NOT_FOUND, f"No job found with id '{job_id}'.",
        status_code=status.HTTP_404_NOT_FOUND,
    )


def job_not_done(job_id: str, status_value: str) -> APIError:
    return APIError(
        JOB_NOT_DONE, f"Job '{job_id}' is not done yet (status: {status_value}).",
        status_code=status.HTTP_409_CONFLICT,
    )


def unauthorized() -> APIError:
    return APIError(
        UNAUTHORIZED, "Missing or invalid session. Please log in.",
        status_code=status.HTTP_401_UNAUTHORIZED,
    )


def invalid_credentials() -> APIError:
    # Deliberately the same message whether the email doesn't exist or
    # the password is wrong -- distinguishing the two in the response
    # would let an attacker enumerate registered emails one guess at a
    # time.
    return APIError(
        INVALID_CREDENTIALS, "Incorrect email or password.",
        status_code=status.HTTP_401_UNAUTHORIZED,
    )


def email_already_registered() -> APIError:
    return APIError(
        EMAIL_ALREADY_REGISTERED, "An account with this email already exists.",
        status_code=status.HTTP_409_CONFLICT,
    )


def forbidden(job_id: str) -> APIError:
    return APIError(
        FORBIDDEN, f"Job '{job_id}' does not belong to the current user.",
        status_code=status.HTTP_403_FORBIDDEN,
    )


def rate_limit_exceeded(daily_limit: int) -> APIError:
    return APIError(
        RATE_LIMIT_EXCEEDED,
        f"Daily limit of {daily_limit} research jobs reached. Please try again tomorrow.",
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
    )


def ticker_not_found(ticker: str) -> APIError:
    return APIError(
        TICKER_NOT_FOUND,
        f"'{ticker}' doesn't look like a valid, currently-listed ticker.",
        status_code=status.HTTP_400_BAD_REQUEST,
    )
