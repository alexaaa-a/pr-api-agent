from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict
import re
import json


model_id = "ibm-granite/granite-4.0-h-350m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

coffee_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=80,
    temperature=0.1,
)

llm = HuggingFacePipeline(pipeline=coffee_pipeline)


class CoffeeState(TypedDict):
    """Состояние"""
    customer_review: str
    emotion: str
    urgency: str
    detected_issue: str
    response_template: str


@tool
def generate_response(emotion: str, urgency: str) -> str:
    """Генерирует ответ для клиента"""
    if urgency == "HIGH":
        return "🚨 URGENT! Customer reported health issues. Contact manager immediately. Offer medical assistance."
    if emotion == "POSITIVE":
        return "😊 Response: 'Thank you for your feedback! We're glad you enjoyed it. Hope to see you again!'"
    elif emotion == "NEGATIVE":
        return "😔 Response: 'We apologize for the inconvenience. We will look into this situation. Please offer a free coffee next time.'"
    else:
        return "😐 Response: 'Thank you for your feedback! We will consider your comments.'"


def analyze_review_node(state: CoffeeState):
    """Анализ отзыва посетителя"""
    review = state["customer_review"]
    prompt = f"""
    Review is in English:
    "{review}"

    Task: Identify emotion, urgency, and possible issue.
    Return only JSON with keys: emotion (POSITIVE/NEGATIVE/NEUTRAL), urgency (LOW/MEDIUM/HIGH), issue (short description)
    """
    try:
        granite_output = coffee_pipeline(prompt)[0]["generated_text"]
        json_match = re.search(r'\{.*\}', granite_output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            emotion = data.get("emotion", "").upper()
            urgency = data.get("urgency", "").upper()
            issue = data.get("issue", "")
        else:
            raise ValueError("No JSON found")
    except Exception:
        review_lower = review.lower()
        if any(word in review_lower for word in ['terrible', 'awful', 'bad', 'horrible', 'disappointing', 'burnt']):
            emotion = "NEGATIVE"
        elif any(word in review_lower for word in ['good', 'great', 'excellent', 'amazing', 'love', 'perfect']):
            emotion = "POSITIVE"
        else:
            emotion = "NEUTRAL"

        if any(word in review_lower for word in ['allergy', 'nausea', 'sick', 'urgent', 'dangerous', 'doctor', 'pain', 'hospital', 'emergency']):
            urgency = "HIGH"
        elif any(word in review_lower for word in ['disappointed', 'terrible', 'nightmare', 'trash', 'bad', 'awful', 'horrible']):
            urgency = "MEDIUM"
        else:
            urgency = "LOW"

        if "bitter" in review_lower:
            issue = "Bitter coffee"
        elif "cold" in review_lower:
            issue = "Cold drink"
        elif "long" in review_lower or "wait" in review_lower:
            issue = "Long waiting time"
        elif "barista" in review_lower and ("bad" in review_lower or "rude" in review_lower):
            issue = "Service problem"
        elif any(word in review_lower for word in ['trash', 'terrible', 'awful', 'horrible']):
            issue = "General quality dissatisfaction"
        elif any(word in review_lower for word in ['allergy', 'nausea', 'sick']):
            issue = "Health problem"
        else:
            issue = "General feedback"

    return {
        **state,
        "emotion": emotion,
        "urgency": urgency,
        "detected_issue": issue
    }

def generate_template_node(state: CoffeeState):
    """Генерация ответа"""
    response = generate_response.invoke({
        "emotion": state["emotion"],
        "urgency": state["urgency"]
    })
    return {
        **state,
        "response_template": response
    }


def create_coffee_agent():
    """Создание агента"""
    workflow = StateGraph(CoffeeState)
    workflow.add_node("analyze", analyze_review_node)
    workflow.add_node("generate", generate_template_node)
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

coffee_agent = create_coffee_agent()

print("🐥 Coffee Review Analyzer is ready!\n")

while True:
    user_input = input("\n📝 Enter customer review (or 'exit'): ")
    if user_input.lower() == 'exit':
        break
    result = coffee_agent.invoke({"customer_review": user_input})
    print(f"\n📊 Review analysis:")
    print(f"🎭 Emotion: {result['emotion']}")
    print(f"🚨 Urgency: {result['urgency']}")
    print(f"🔧 Detected issue: {result['detected_issue']}")
    print(f"💬 Response template: {result['response_template']}")
    print("=" * 60)
