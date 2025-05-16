# app/services/tools/manage_tasks.py

from typing import Dict, List, Any, Optional, Union
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.task import Task, TaskList

class ManageTasksInput(BaseModel):
    action: str = Field(..., description="Action to perform: 'add' or 'list'")
    task: Optional[Dict[str, str]] = Field(None, description="Task details when adding")

class ManageTasksTool(BaseTool):
    name = "manage_tasks"
    description = "Manages user tasks like scheduling demos and follow-ups"
    args_schema = ManageTasksInput
    
    def __init__(self):
        super().__init__()
        self.tasks: Dict[str, TaskList] = {}
    
    def _get_user_tasks(self, user_id: str) -> TaskList:
        """Get task list for a specific user."""
        if user_id not in self.tasks:
            self.tasks[user_id] = TaskList()
        return self.tasks[user_id]
    
    def _run(self, action: str, task: Optional[Dict[str, str]] = None, user_id: str = "default") -> Union[str, Dict]:
        """Run the tool to manage tasks."""
        user_tasks = self._get_user_tasks(user_id)
        
        if action == "add" and task:
            new_task = Task(**task)
            user_tasks.tasks.append(new_task)
            return f"Task '{new_task.title}' added for {new_task.when}"
        
        elif action == "list":
            return {"tasks": user_tasks.tasks}
        
        else:
            return "Invalid action. Use 'add' or 'list'."