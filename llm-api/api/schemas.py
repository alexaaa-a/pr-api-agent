from pydantic import BaseModel

class Request(BaseModel):
    """Ввод промпта для LLM"""
    message: str
    max_tokens: int = 350

class Response(BaseModel):
    """Ответ LLM"""
    response: str

class TranslateRequest(BaseModel):
    """Ввод запроса на перевод текста"""
    text: str
    target_language: str

class ExplainRequest(BaseModel):
    """Ввод запроса на объяснение концепции"""
    termin: str
    max_tokens: int = 400

class RewriteRequest(BaseModel):
    """Ввод текста для перефразирования"""
    phrase: str