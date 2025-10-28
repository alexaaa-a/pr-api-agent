from transformers import AutoModelForSequenceClassification, pipeline

def load_model():
    """Загрузка модели"""
    model_name = 'jitesh/emotion-english'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classf = pipeline("text-classification", model=model, tokenizer=model_name)
    return classf


classifier = load_model()