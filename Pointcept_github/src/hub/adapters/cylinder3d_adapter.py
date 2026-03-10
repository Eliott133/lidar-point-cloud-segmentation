from pathlib import Path
from typing import Any

from hub.adapters.base import BaseAdapter


class Cylinder3DAdapter(BaseAdapter):
    name = "cylinder3d"

    def __init__(self, repo_path: str = "external/cylinder3d") -> None:
        self.repo_path = Path(repo_path)

    def train(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.name,
            "repo_path": str(self.repo_path),
            "status": "todo",
        }

    def evaluate(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.name,
            "status": "todo",
        }

    def predict(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.name,
            "status": "todo",
        }