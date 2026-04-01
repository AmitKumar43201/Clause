from utils.tokencount import count_tokens
from .systemprompt import get_system_prompt
from dataclasses import dataclass


@dataclass
class MessageItem:
    role: str
    content: str
    tonken_count: int | None = None
    
class ContextManager:
    def __init__(self) -> None:
        self._system_prompt = get_system_prompt()
        self._model_name = "mistralai/devstral-2512:free"
        self._messages: list[MessageItem] = []

    def add_user_message(self, content: str) -> None:
        item = MessageItem(
            role="user",
            content=content,
            token_count=count_tokens(
                content,
                self._model_name,
            ),
        )
        self._messages.append(item)
        
    def add_assistant_message(self, content: str) -> None:
        item = MessageItem(
            role="assistant",
            content=content or "",
            token_count=count_tokens(
                content,
                self._model_name,
            ),
        )
        self._messages.append(item)