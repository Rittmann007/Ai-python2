import os
from fastapi import Header,HTTPException
import secrets

INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY")


# auth function
async def verify_internal_key(x_internal_key: str = Header(...)):
    """
    Verifies the shared key between microservice and main backend
    """
    if not INTERNAL_API_KEY or not secrets.compare_digest(x_internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing internal API key")