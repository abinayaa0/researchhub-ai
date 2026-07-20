from fastapi import FastAPI
from database import Base, engine
from routers import auth, papers

app = FastAPI(title="ResearchHub AI")
#app is an instance of fastAPI class and WE named the instanece app.


Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(papers.router)


@app.get("/health")
def health():
    return {"status": "ok"}