from typing import List
from fastapi import APIRouter
from .func import store_data

def request(router: APIRouter, pharse_id: int=1, prob_id: int=1)->None:
    @router.post(f"/prob-{prob_id}/predict")
    async def run(id: int, columns: List[str], rows: List[List[float]]):
        return await store_data(
            id=id,
            columns=columns,
            rows=rows, 
            pharse=f"pharse-{pharse_id}",
            problem=f"prob-{prob_id}"
        )
        
def main_router(pharse_id:int=1, prob_ids:List[int]=[1,2])->APIRouter:
    router = APIRouter(
        prefix=f"/pharse-{pharse_id}",
        tags=["models"],
        responses={404: {"description": "Not found"}},
    )

    for prob_id in prob_ids:
        request(router, pharse_id, prob_id)

    return router


