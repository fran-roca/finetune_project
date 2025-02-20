import torch
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from typing import Dict, Any
from src.utils.logger import setup_logger
from src.utils.device import get_device

logger = setup_logger(__name__)

class ModelFactory:
    @staticmethod
    def create(model_config: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        """
        Loads the model specified in model_config, resizes its token embeddings,
        and moves it to the appropriate device (using CUDA if available).

        Args:
            model_config (Dict[str, Any]): Dictionary with model parameters, including 'name'.
            tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.

        Returns:
            PreTrainedModel: The loaded and configured model.
        """
        model_name: str = model_config["name"]
        device = get_device()
        logger.info("Detected device: %s", device)
        device_map: str = "auto" if device == "cuda" else None
        logger.info("Loading model: %s", model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation='eager',
            device_map=device_map
        )
        model.resize_token_embeddings(len(tokenizer))
        if device == "cuda":
            model.to(torch.bfloat16)
        else:
            model.to(device)
        return model
