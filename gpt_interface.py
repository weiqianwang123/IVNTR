import os
import openai
from typing import List, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt

class GPTInterface:
    """A simple interface to interact with OpenAI's GPT models."""
    
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize the GPT interface.
        
        Args:
            model_name: The name of the GPT model to use (default: "gpt-4")
        """
        self.model_name = model_name
        self._set_api_key()
        
    def _set_api_key(self, key: Optional[str] = None) -> None:
        """Set the OpenAI API key."""
        if key is None:
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            key = os.environ["OPENAI_API_KEY"]
        openai.api_key = key

    @retry(wait=wait_random_exponential(min=1, max=60),
           stop=stop_after_attempt(10))
    def generate_response(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        seed: Optional[int] = None
    ) -> str:
        """Generate a response from GPT.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens in the response
            seed: Random seed for reproducibility
            
        Returns:
            The generated response text
        """
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed
        )
        
        if not completion.choices:
            raise ValueError("No response received from GPT")
            
        return completion.choices[0].message.content

    def chat(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        seed: Optional[int] = None
    ) -> str:
        """A convenience method for simple chat interactions.
        
        Args:
            prompt: The user's message
            system_message: Optional system message to set context
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens in the response
            seed: Random seed for reproducibility
            
        Returns:
            The generated response text
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        return self.generate_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed
        ) 