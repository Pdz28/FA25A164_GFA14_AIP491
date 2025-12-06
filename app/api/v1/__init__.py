"""API v1 router aggregation."""
from fastapi import APIRouter

from app.api.v1 import health, predict

router = APIRouter()
router.include_router(health.router)
router.include_router(predict.router)
