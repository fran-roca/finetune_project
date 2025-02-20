from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Any
import torch
from src.utils.logger import setup_logger
from src.utils.device import get_device

logger = setup_logger(__name__)

def evaluate_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, max_new_tokens: int = 300) -> str:
    """
    Generates a response from the model based on the given prompt.

    Args:
        model (PreTrainedModel): The fine-tuned model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        prompt (str): The input text prompt.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The decoded output from the model.
    """
    device = get_device()
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.01,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0])
    logger.info("Evaluation complete.")
    return result
