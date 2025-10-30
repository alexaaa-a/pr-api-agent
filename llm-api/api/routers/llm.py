from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenModel:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        print("Загружаю модель...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Модель загружена!")

    def generate_response(self, message, max_new_tokens=350):
        messages = [
            {"role": "user", "content": message},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        return response


qwen_model = QwenModel()