from trl import SFTTrainer, SFTConfig
from typing import Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Any,
    eval_dataset: Any,
    lora_config: Any,
    training_args: Dict[str, Any]
) -> SFTTrainer:
    """
    Creates and returns an SFTTrainer instance configured for training.

    Args:
        model (PreTrainedModel): The model to fine-tune.
        tokenizer (PreTrainedTokenizer): The tokenizer used for preprocessing.
        train_dataset (Any): Training dataset.
        eval_dataset (Any): Evaluation dataset.
        lora_config (Any): LoRA configuration.
        training_args (Dict[str, Any]): Dictionary of training hyperparameters.

    Returns:
        SFTTrainer: The trainer instance.
    """
    trainer_config = SFTConfig(**training_args)
    trainer = SFTTrainer(
        model=model,
        args=trainer_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    logger.info("Trainer created with configuration: %s", trainer_config)
    return trainer
