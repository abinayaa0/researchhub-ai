import os
import sys
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# database.py expects DATABASE_URL directly; construct it here since
# it's normally only built by docker-compose's variable substitution
os.environ["DATABASE_URL"] = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@localhost:5432/{os.getenv('POSTGRES_DB')}"
)
print(f"DEBUG - Using DATABASE_URL: {os.environ['DATABASE_URL']}")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import User, Paper

OWNER_EMAIL = "user@example.com"

PAPERS = [
    ("Attention Is All You Need", "attention_is_all_you_need.pdf", "paper-attention"),
    ("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "bert.pdf", "paper-bert"),
    ("RoBERTa: A Robustly Optimized BERT Pretraining Approach", "roberta.pdf", "paper-roberta"),
    ("Language Models are Unsupervised Multitask Learners", "gpt2.pdf", "paper-gpt2"),
    ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "vit.pdf", "paper-vit"),
    ("Deep Residual Learning for Image Recognition", "resnet.pdf", "paper-resnet"),
]

engine = create_engine(os.environ["DATABASE_URL"])
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

user = db.query(User).filter(User.email == OWNER_EMAIL).first()
if not user:
    print(f"No user found with email {OWNER_EMAIL} — register this user first via /auth/register")
    sys.exit(1)

print(f"Found user: {user.email} (id={user.id})")

for title, filename, paper_id in PAPERS:
    existing = db.query(Paper).filter(Paper.qdrant_paper_id == paper_id).first()
    if existing:
        print(f"  Already exists: {title}")
        continue

    paper = Paper(
        user_id=user.id,
        title=title,
        filename=filename,
        storage_path=f"eval/corpus_papers/{filename}",
        status="ready",
        qdrant_paper_id=paper_id,
    )
    db.add(paper)
    print(f"  Created: {title}")

db.commit()
print("\nDone.")