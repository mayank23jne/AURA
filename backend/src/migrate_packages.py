from sqlalchemy import text, inspect
from src.db import get_engine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate():
    engine = get_engine()
    
    with engine.connect() as conn:
        # 1. Create packages table
        logger.info("Checking/Creating 'packages' table...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS packages (
                id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # 2. Add package_id to policies table
        logger.info("Checking/Updating 'policies' table...")
        
        # Check if column exists (MySQL specific logic, but works for general SQL check often)
        # Using SQLAlchemy inspector for DB-agnostic check would be better, but direct SQL is faster for now
        inspector = inspect(engine)
        columns = [c['name'] for c in inspector.get_columns('policies')]
        
        if 'package_id' not in columns:
            logger.info("Adding 'package_id' column to policies...")
            conn.execute(text("ALTER TABLE policies ADD COLUMN package_id VARCHAR(50)"))
            
            # Check dialect
            if engine.dialect.name != 'sqlite':
                conn.execute(text("ALTER TABLE policies ADD CONSTRAINT fk_policy_package FOREIGN KEY (package_id) REFERENCES packages(id)"))
            else:
                logger.info("Skipping FK constraint for SQLite (not supported in ALTER TABLE easily)")
        else:
            logger.info("'package_id' column already exists.")

        conn.commit()
        logger.info("Migration completed successfully.")

if __name__ == "__main__":
    migrate()
