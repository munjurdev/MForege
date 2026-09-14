"""
Conversation Memory System
==========================
Manages conversation history with support for different message types.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from datetime import datetime


def _default_timestamp():
    return datetime.now()


class Message(BaseModel):
    """Represents a single message in the conversation"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    timestamp: datetime = Field(default_factory=_default_timestamp)
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


class ConversationMemory:
    """
    Manages conversation history.
    
    Features:
    - Stores messages with timestamps
    - Limits maximum conversation length
    - Can export in OpenAI format
    - Supports clearing and truncation
    """
    
    def __init__(self, max_messages: int = 100):
        self.messages: list[Message] = []
        self.max_messages = max_messages
        
    def add(self, message: Message) -> None:
        """Add a message to history"""
        self.messages.append(message)
        
        # Trim old messages if exceeding limit (keep system messages)
        if len(self.messages) > self.max_messages:
            self._trim()
            
    def _trim(self) -> None:
        """Trim messages to stay within limits, preserving important messages"""
        # Keep system messages and recent messages
        system_msgs = [m for m in self.messages if m.role == "system"]
        other_msgs = [m for m in self.messages if m.role != "system"]
        
        # Keep last N non-system messages (guard against negative slice sizes)
        keep_count = max(0, self.max_messages - len(system_msgs))
        kept_other = other_msgs[-keep_count:] if keep_count > 0 else []
        self.messages = system_msgs + kept_other
        
    def get(self, role: Optional[str] = None, limit: Optional[int] = None) -> List[Message]:
        """Get messages, optionally filtered by role and limited"""
        result = self.messages
        
        if role:
            result = [m for m in result if m.role == role]
            
        if limit:
            result = result[-limit:]
            
        return result
        
    def to_openai_format(self) -> List[Dict]:
        """Convert messages to OpenAI API format"""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
        ]
        
    def to_dict_list(self) -> List[Dict]:
        """Convert all messages to list of dicts"""
        return [m.to_dict() for m in self.messages]
        
    def clear(self) -> None:
        """Clear all messages except system messages"""
        self.messages = [m for m in self.messages if m.role == "system"]
        
    def last_message(self) -> Optional[Message]:
        """Get the last message"""
        return self.messages[-1] if self.messages else None
        
    @property
    def count(self) -> int:
        """Get total message count"""
        return len(self.messages)
        
    @property
    def last_user_message(self) -> Optional[str]:
        """Get the last user message content"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None
