from fastapi import APIRouter, HTTPException, status
from .llm import qwen_model
from .. import schemas

router = APIRouter()

@router.get("/")
async def root():
    """Корневой эндпоинт для проверки работы API"""
    return {"message": "LLM API is running"}

@router.post("/chat")
async def chat(request: schemas.Request):
    """Чат-эндпоинт для промптов"""
    try:
        response = qwen_model.generate_response(
            message=request.message,
            max_new_tokens=request.max_tokens
        )

        return schemas.Response(
            response=response,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM API failed: {str(e)}"
        )

@router.post("/translate")
async def translate(request: schemas.TranslateRequest):
    """Эндпоинт для запроса на перевод текста"""
    try:
        data = f"Переведи фразу '{request.text}' на {request.target_language} язык"
        response = qwen_model.generate_response(
            message=data,
        )

        return schemas.Response(
            response=response,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM API failed: {str(e)}"
        )

@router.post("/explain")
async def explain(request: schemas.ExplainRequest):
    """Эндпоинт для объяснения термина/концепции"""
    try:
        data = f"Кратко объясни понятие: {request.termin}"
        response = qwen_model.generate_response(
            message=data,
            max_new_tokens=request.max_tokens
        )

        return schemas.Response(
            response=response,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM API failed: {str(e)}"
        )

@router.post("/rewrite")
async def rewrite(request: schemas.RewriteRequest):
    """Эндпоинт для перефразирования текста"""
    try:
        data = f"Перефразируй текст кратко: '{request.phrase}'"
        response = qwen_model.generate_response(
            message=data,
        )

        return schemas.Response(
            response=response,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM API failed: {str(e)}"
        )
