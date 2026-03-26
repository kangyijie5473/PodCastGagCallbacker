from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os

class LLMModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate response from LLM.
        """
        pass

class OpenAILLM(LLMModel):
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        from openai import OpenAI
        from dotenv import load_dotenv  # 加载.env文件
        load_dotenv()
        # Try to get from env if not provided
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL", "../mle_train/deepseek-14B-awq/")
        if not self.api_key:
             # If no key, we might be using a local server that doesn't strictly require one,
             # but OpenAI client usually wants something.
             self.api_key = "sk-placeholder" 

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        print("base url:" + self.base_url)

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
