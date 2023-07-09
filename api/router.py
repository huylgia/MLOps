from typing import List, Any
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from tools.predict import Predictor
from .func import predict, store_data

class Item(BaseModel):
    id: str
    columns: List[str]
    rows: List[List[Any]]

def request(router: APIRouter, predictor: Predictor, phase_id: int=1, prob_id: int=1)->None: 
    @router.post(f"/prob-{prob_id}/predict")
    async def run(item: Item, background_tasks: BackgroundTasks):
        background_tasks.add_task(
            store_data, 
            **item.__dict__,
            phase=f"phase-{phase_id}",
            problem=f"prob-{prob_id}"
        )

        return await predict(
            **item.__dict__,
            predictor=predictor,
        )
        
def main_router(phase_id:int=1, prob_ids:List[int]=[1,2])->APIRouter:
    router = APIRouter(
        prefix=f"/phase-{phase_id}",
        tags=["models"],
        responses={404: {"description": "Not found"}},
    )

    for prob_id in prob_ids:
        predictor = Predictor(phase=f"/phase-{phase_id}", problem=f"prob-{prob_id}", postfix="_2")
        request(
            router,
            predictor=predictor, 
            phase_id=phase_id,
            prob_id=prob_id
        )

    return router


