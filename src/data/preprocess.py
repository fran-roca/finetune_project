from typing import Any, Dict
from transformers import PreTrainedTokenizer

def preprocess_sample(sample: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> Dict[str, str]:
    """
    Processes a single data sample by applying the chat template.

    If the first message has the role 'system', its content is merged with the subsequent message.

    Args:
        sample (Dict[str, Any]): A dictionary representing a data sample with a "messages" key.
        tokenizer (PreTrainedTokenizer): The tokenizer with a custom chat_template.

    Returns:
        Dict[str, str]: A dictionary with a "text" key containing the processed prompt.
    """
    messages = sample.get("messages", [])
    if messages and messages[0].get("role") == "system" and len(messages) > 1:
        system_content = messages[0]["content"]
        messages[1]["content"] = system_content + "\n" + messages[1]["content"]
        messages = messages[1:]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": prompt}
