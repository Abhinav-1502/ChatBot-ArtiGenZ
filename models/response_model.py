from pydantic import BaseModel, Field
from typing import List, Optional, Union, Any

class MetaData(BaseModel):
    sql_query: str = "N/A"
    context_pages: List[dict] = []
    rewritten_query: str = ""
    raw_error: Optional[Union[str, dict]] = None

class ChatResponse(BaseModel):
    status: str  # "success" or "error"
    source: str  # "sql", "rag", "classifier", "unknown"
    message: str
    bot_response: Any
    meta: MetaData