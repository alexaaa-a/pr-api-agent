from pydantic import BaseModel

class Text(BaseModel):
    """Ввод текста для модели"""
    text: str

class Prediction(BaseModel):
    """Формат вывода предсказания модели"""
    request: str
    prediction: str
    confidence: float

class RequestBatch(BaseModel):
    """Ввод списка текстов для модели"""
    request: list[str]
