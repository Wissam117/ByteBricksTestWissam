# app/models/task.py

from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class Task(BaseModel):
    title: str
    when: str
    description: str
    created_at: Optional[datetime] = None
    
    def __init__(self, **data):
        if "created_at" not in data:
            data["created_at"] = datetime.now()
        super().__init__(**data)
        
class TaskList(BaseModel):
    tasks: List[Task] = []