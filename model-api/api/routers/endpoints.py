from fastapi import APIRouter, HTTPException, status
from .model import classifier
from .. import schemas

router = APIRouter()

@router.get("/")
async def root():
    """Корневой эндпоинт для проверки работы API"""
    return {"message": "Emotion detection API is running"}

@router.post("/predict")
async def predict(text: schemas.Text):
    """Получить предсказание модели по одному тексту"""
    try:
        prediction = classifier.predict(text.text)
        return schemas.Prediction(
            request=text.text,
            prediction=prediction[0]['label'],
            confidence=prediction[0]['score']
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/predict_batch")
async def predict_batch(batch: schemas.RequestBatch):
    """Получить предсказание модели по нескольким текстам (список)"""
    response = {}
    lst = batch.request
    try:
        for req in range(len(lst)):
            prediction = classifier.predict(lst[req])
            response[f'prediction_{req}'] = schemas.Prediction(
                request=lst[req],
                prediction=prediction[0]['label'],
                confidence=prediction[0]['score']
            )

        return response


    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/info")
async def info():
    """Получить информацию о модели"""
    return {"message": {
        "architecture": "transformer-based",
        "task": "emotion detection",
        "language": "english",
        "number of emotion labels": 20
    }}

@router.get("/examples")
async def examples():
    """Получить примеры входных и выходных данных модели"""
    return {"message": {
        "example_1": {
            "request": "This is completely unacceptable and I will not stand for it any longer!",
            "prediction": "anger",
            "confidence": 0.9884265661239624
        },
        "example_2": {
            "request": "This is the most wonderful news I have heard all year!",
            "prediction": "joy",
            "confidence": 0.9880096316337585
        },
        "example_3": {
            "request": "That seems like an oddly convenient excuse, if you ask me.",
            "prediction": "cheeky",
            "confidence": 0.38162875175476074
        },
        "example_4": {
            "request": "I cannot believe I forgot your birthday; I feel just awful about it.",
            "prediction": "guilty",
            "confidence": 0.9750471115112305
        },
        "example_5": {
            "request": "So, how exactly does that work? I would love to understand the details.",
            "prediction": "curious",
            "confidence": 0.9648261070251465
        }
    }}

@router.get("/health")
async def health():
    """Проверка состояния модели"""
    try:
        pred = classifier.predict("This is the most wonderful news I have heard all year!")
        return {"message": "Model is healthy!"}


    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model isn't healthy!"
        )
