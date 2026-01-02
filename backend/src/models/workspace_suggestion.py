from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class WorkspaceSuggestion(Base):
    __tablename__ = "workspace_suggestions"

    id = Column(String(50), primary_key=True)  # use timestamp string ID
    title = Column(String(255), nullable=False)
    description = Column(String(1024), nullable=False)
    resources = Column(JSON, nullable=False)  # list of resource dicts
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    dismissed = Column(Boolean, default=False)
    accepted = Column(Boolean, default=False)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "resources": self.resources,
            "created_at": self.created_at.isoformat(),
            "dismissed": self.dismissed,
            "accepted": self.accepted,
        }
