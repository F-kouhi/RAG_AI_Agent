from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

API_KEY = os.getenv('EMBEDDING_API_KEY')

class OpenRouterLLM:
    def __init__(
        self,
        api_key: str = API_KEY,
        model: str = "gapgpt-qwen-3.5",
        base_url: str = "https://api.gapgpt.app/v1",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        summary: bool = True,
        system_prompt:tuple = ("شما یک دستیار فارسی هستید که پاسخ‌ها را به صورت خلاصه اما دقیق ارائه می‌دهد.",),
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.summary = summary
        self.system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        if self.summary:
            system_content = self.system_prompt
        else:
            system_content = self.system_prompt

        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content.strip()
