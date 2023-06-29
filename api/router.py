from typing import List
from fastapi import APIRouter
from pydantic import BaseModel

from .func import store_data

class Item(BaseModel):
    id: str
    columns: List[str]
    rows: List[List[float]]

def request(router: APIRouter, phase_id: int=1, prob_id: int=1)->None:
    @router.post(f"/prob-{prob_id}/predict")
    async def run(item: Item):
        return await store_data(
            **item.__dict__,
            phase=f"phase-{phase_id}",
            problem=f"prob-{prob_id}"
        )
        
def main_router(phase_id:int=1, prob_ids:List[int]=[1,2])->APIRouter:
    router = APIRouter(
        prefix=f"/phase-{phase_id}",
        tags=["models"],
        responses={404: {"description": "Not found"}},
    )

    for prob_id in prob_ids:
        request(router, phase_id, prob_id)

    return router


