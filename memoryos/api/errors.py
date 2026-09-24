from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Raw rejected input can contain nonfinite floats that JSON cannot represent.
    errors = [{key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})
