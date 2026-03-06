from abc import ABC, abstractmethod
from typing import Any


class BaseAdapter(ABC):
    name: str

    @abstractmethod
    def train(self, config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError