from fastapi import FastAPI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Resume Tool API")

@app.get("/")
async def root():
    return {"message": "AI Resume Tool API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=PORT)