from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config.settings import get_settings
import logging
import os

import pymysql         # add this
pymysql.install_as_MySQLdb()   # add this

logger = logging.getLogger(__name__)
settings = get_settings()

def get_engine():
    # Enforce MySQL usage with pymysql as requested
    mysql_url = settings.database.mysql_url
    
    # Ensure URL uses pymysql driver
    if "mysql" in mysql_url and "pymysql" not in mysql_url:
        mysql_url = mysql_url.replace("mysql+mysqlconnector:", "mysql+pymysql:")
        if "pymysql" not in mysql_url and "mysql:" in mysql_url:
             mysql_url = mysql_url.replace("mysql:", "mysql+pymysql:")

    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Connecting to MySQL: {mysql_url} (Attempt {attempt+1}/{max_retries})...")
            engine = create_engine(
                mysql_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={"connect_timeout": 5}
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("Connected to MySQL database")
            return engine
        except Exception as e:
            print(f"MySQL connection failed (Attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    print("All MySQL connection attempts failed. Falling back to SQLite")
    sqlite_url = "sqlite:///./aura.db"
    engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False}
    )
    print(f"Connected to SQLite database: {sqlite_url}")
    return engine

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    from src.models.orm import Base
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized")

# Initialize tables on import (or you can call this explicitly in main.py)
if __name__ == "__main__":
    init_db()
