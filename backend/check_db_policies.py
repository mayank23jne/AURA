import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.dirname(__file__))

from src.db import SessionLocal
from src.models.orm import ORMPolicy

def check_policies():
    db = SessionLocal()
    try:
        count = db.query(ORMPolicy).count()
        print(f"Total policies in DB: {count}")
        policies = db.query(ORMPolicy).all()
        for p in policies:
            print(f"- {p.id}: {p.name}")
    except Exception as e:
        print(f"Error querying DB: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    check_policies()
