from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict


model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.2,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)


class SimpleLLMState(TypedDict):
    user_input: str
    result: str


@tool
def math_solver(expression: str) -> str:
    """Решает арифметику через LLM"""
    prompt = f"Calculate the result of this expression: {expression}. Give only the answer and nothing more."
    answer = llm.invoke(prompt).strip()
    lines = answer.split("\n")[1:]
    for line in lines:
        if 'answer' in line:
            return line
    return answer

@tool
def translator(text: str) -> str:
    """Переводчик через LLM"""
    prompt = f"Translate the text into German: '{text}'. Just give me the translated text"
    answer = llm.invoke(prompt).strip()
    lines = answer.split("\n")[1:]
    for line in range(len(lines)):
        if 'answer' or 'transl' in lines[line]:
            return lines[line + 1]
    return answer


def llm_agent_node(state: SimpleLLMState):
    text = state["user_input"]

    if any(c.isdigit() for c in text):
        result = math_solver.invoke({"expression": text})
    else:
        result = translator.invoke({"text": text})

    return {**state, "result": result}


def create_llm_agent():
    graph = StateGraph(SimpleLLMState)
    graph.add_node("process", llm_agent_node)
    graph.set_entry_point("process")
    graph.add_edge("process", END)
    return graph.compile()

agent = create_llm_agent()


print("🧮 LLM Math & Translator Agent is ready!\n")

while True:
    user_input = input("\nВведите выражение или текст (или 'exit'): ")
    if user_input.lower() == "exit":
        break
    result = agent.invoke({"user_input": user_input})
    print("✅ Result:", result["result"])
