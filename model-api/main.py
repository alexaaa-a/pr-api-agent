from fastapi import FastAPI
from api.routers import endpoints

app = FastAPI()

app.include_router(endpoints.router)