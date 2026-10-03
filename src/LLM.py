from transformers import AutoTokenizer, AutoModelForCausalLM

class llm_model:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    def Generate_answer(self, query: str, context):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions "
                    "using the provided context from the user's files.\n"
                    "Only use information from the provided context. "
                    'If the context does not contain enough information, say "I don\'t know."'
                )
            },
            {
                "role": "user",
                "content": (
                    f"<context>\n"
                    f"{context}\n"
                    f"</context>\n\n"
                    f"<question>\n"
                    f"{query}\n"
                    f"</question>"
                )
            }
        ]

        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,

        )
        output = self.model.generate(
            **ids,
            max_new_tokens=1000,
            
        )
        return self.tokenizer.decode(
            output.tolist()[0][len(ids["input_ids"][0]):],
            skip_special_tokens=True
        )

