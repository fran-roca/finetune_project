from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Any, Dict, Optional
import torch
from src.utils.logger import setup_logger
from src.utils.device import get_device

logger = setup_logger(__name__)

def evaluate_model(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    prompt: str, 
    generation_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generates a response from the model based on the given prompt using configurable generation parameters.

    Args:
        model (PreTrainedModel): The fine-tuned model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        prompt (str): The input text prompt.
        generation_params (Optional[Dict[str, Any]]): Override default generation parameters.

    Returns:
        str: The decoded output from the model.

    Raises:
        ValueError: If generation fails.
    """
    device = get_device()
    default_params = {
        "max_new_tokens": 300,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.01,
        "repetition_penalty": 1.0,
    }
    if generation_params:
        default_params.update(generation_params)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **default_params,
                eos_token_id=tokenizer.eos_token_id
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=False)
        logger.info("Evaluation complete.")
        return result
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise ValueError(f"Evaluation failed: {str(e)}")
