from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
app = FastAPI()
sentiment_analyzer = pipeline("sentiment-analysis")
class TextRequest(BaseModel):
    text: str
    
@app.post("/process_text")
def process_text(request: TextRequest):
    result = sentiment_analyzer(request.text)[0]
    return {"sentiment": result['label'], "score": result['score']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)