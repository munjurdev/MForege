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
    - Auto-condensation: when history approaches the model's context
      window, older turns are folded into a compact summary (like an
      agent session summary) instead of being silently dropped
    """
    
    # Condense when estimated usage crosses this fraction of the window
    CONDENSE_AT = 0.70
    # Recent messages always kept verbatim when condensing
    KEEP_RECENT = 8
    # Max chars of each folded message kept in the summary
    SUMMARY_CLIP = 160

    def __init__(self, max_messages: int = 100):
        self.messages: list[Message] = []
        self.max_messages = max_messages
        self.summary: str = ""  # condensed memory of folded older turns
        
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
        
    def token_estimate(self) -> int:
        """Rough token count (chars/4 heuristic — good enough for a meter)."""
        total = sum(len(m.content) for m in self.messages)
        if self.summary:
            total += len(self.summary)
        return total // 4

    def maybe_condense(self, context_window: int) -> bool:
        """
        Fold the oldest turns into `summary` when history grows past
        CONDENSE_AT of the context window. Returns True if a fold happened.

        Mirrors how agent sessions survive long runs: nothing important is
        silently dropped — the oldest parts become a compact digest that is
        injected as a system message, recent turns stay verbatim.
        """
        if context_window <= 0:
            return False
        if self.token_estimate() < int(context_window * self.CONDENSE_AT):
            return False
        system_msgs = [m for m in self.messages if m.role == "system"]
        other_msgs = [m for m in self.messages if m.role != "system"]
        if len(other_msgs) <= self.KEEP_RECENT:
            return False

        fold = other_msgs[:-self.KEEP_RECENT]
        lines = []
        for m in fold:
            tag = {"user": "User", "tool": "Tool"}.get(m.role, "MForege")
            text = " ".join(m.content.split())
            if text:
                lines.append(f"- {tag}: {text[:self.SUMMARY_CLIP]}")
        if lines:
            self.summary = (self.summary + "\n" if self.summary else "") + "\n".join(lines)

        keep_count = max(0, self.max_messages - len(system_msgs))
        self.messages = system_msgs + other_msgs[-self.KEEP_RECENT:][-keep_count:]
        return True

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
