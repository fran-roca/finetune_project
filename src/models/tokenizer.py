from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import List, Dict, Any

DEFAULT_SPECIAL_TOKENS: List[str] = [
    "<pad>", "<eos>", "<tools>", "</tools>",
    "<think>", "</think>", "<tool_call>",
    "</tool_call>", "<tool_response>", "</tool_response>"
]

DEFAULT_CHAT_TEMPLATE: str = (
    "{{ bos_token }}{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}{% endif %}"
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] | trim + '<end_of_turn><eos>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}"
)

class TokenizerFactory:
    @staticmethod
    def create(model_config: Dict[str, Any]) -> PreTrainedTokenizer:
        """
        Creates and returns a tokenizer using parameters specified in model_config.
        If a key is not provided, default values are used.

        Args:
            model_config (Dict[str, Any]): Dictionary with model parameters including:
                - name: The model name or path.
                - pad_token: (Optional) Token to use for padding.
                - chat_template: (Optional) Custom chat template string.
                - special_tokens: (Optional) List of special tokens to add.

        Returns:
            PreTrainedTokenizer: The configured tokenizer.
        """
        model_name: str = model_config["name"]
        special_tokens: List[str] = model_config.get("special_tokens", DEFAULT_SPECIAL_TOKENS)
        pad_token: str = model_config.get("pad_token", special_tokens[0])
        chat_template: str = model_config.get("chat_template", DEFAULT_CHAT_TEMPLATE)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            additional_special_tokens=special_tokens
        )
        tokenizer.pad_token = pad_token
        tokenizer.chat_template = chat_template
        return tokenizer
