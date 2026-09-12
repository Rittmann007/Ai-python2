from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from fastapi import FastAPI,Depends
from ai_python2.Controllers import ingestController,chatController
from ai_python2.Auth import verify_internal_key
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
from langgraph.checkpoint.memory import InMemorySaver


uri = os.environ.get("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi("1"))

@asynccontextmanager # this section will run before the app actually starts
async def lifespan(app: FastAPI):
    app.state.mongo_client = client # store the Mongo client on the app during lifespan, then read it inside the controller.
    app.state.checkpointer = InMemorySaver() # session memory
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!", flush=True)
    except Exception as e:
        print(e, flush=True)
    yield

app = FastAPI(lifespan=lifespan)


#routes

# root
@app.get("/")
async def root():
    return {"message": "Hello World"}

# ingest
app.post("/ingest", dependencies=[Depends(verify_internal_key)])(ingestController)

# chat
app.post("/chat", dependencies=[Depends(verify_internal_key)])(chatController)

# health
@app.get("/health") # for render health check
def health():
    return {"status": "ok"}

