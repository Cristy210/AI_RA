from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

class MLXModel:
    def __init__(
        self,
        model_name: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        max_tokens:int = 700,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens

        self.sampler = make_sampler(temp=0.5, top_k=20)

        print(f"Loading MLX Model: {self.model_name}")
        self.model, self.tokenizer = load(self.model_name)
    
    def generate_response(self, prompt:str) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        response = generate(
            self.model,
            self.tokenizer,
            prompt=chat_prompt,
            max_tokens = self.max_tokens,
            sampler = self.sampler,
            verbose=False,
        )
        return response
