from fastapi import FastAPI
import uvicorn
from api import main_router 

app = FastAPI()
app.include_router(main_router(phase_id=1, prob_ids=[1,2]))
app.include_router(main_router(phase_id=2, prob_ids=[1,2]))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        port=1234
    )