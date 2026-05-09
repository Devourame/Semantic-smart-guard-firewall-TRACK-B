from fastapi import FastAPI
from pydantic import BaseModel

from app.classifier import ai_classifier

app = FastAPI()


class PromptRequest(BaseModel):
    prompt: str


@app.get("/")
def home():

    return {
        "message": "SmartGuard API Running"
    }


@app.post("/classify")
def classify(req: PromptRequest):

    return ai_classifier(req.prompt)