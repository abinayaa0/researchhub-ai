from fastapi import FastAPI
from database import Base, engine
from routers import auth, papers, query

app = FastAPI(title="ResearchHub AI")

Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(papers.router)
app.include_router(query.router)


@app.get("/health")
def health():
    return {"status": "ok"}