from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn
from api import main_router 
from utils import build_logger

LOGGER = build_logger("logger", "error")

app = FastAPI()
# app.include_router(main_router(phase_id=1, prob_ids=[1,2]))
app.include_router(main_router(phase_id=2, prob_ids=[1,2]))

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	LOGGER.info(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=1234,
        workers=4
    )