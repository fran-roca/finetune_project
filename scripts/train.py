#!/usr/bin/env python
import argparse
import logging
from datasets import load_dataset

from src.utils.config_loader import ConfigLoader
from src.utils.config_validator import validate_config
from src.utils.logger import setup_logger
from src.models.tokenizer import TokenizerFactory
from src.models.model_factory import ModelFactory
from src.models.lora_config import get_lora_config
from src.data.preprocess import preprocess_sample
from src.training.trainer_factory import create_trainer

def main(config_path: str) -> None:
    try:
        logger = setup_logger("train", logging.INFO, "logs/train.log")
        config = ConfigLoader.load(config_path)
        
        # Validate configuration
        errors = validate_config(config)
        if errors:
            for err in errors:
                logger.error(err)
            raise ValueError("Configuration validation failed.")
        
        # Create tokenizer and model using configuration
        model_config = config["model"]
        tokenizer = TokenizerFactory.create(model_config)
        model = ModelFactory.create(model_config, tokenizer)
        
        # Load and preprocess dataset
        dataset = load_dataset(config["dataset_name"])
        dataset = dataset.map(
            lambda sample: preprocess_sample(sample, tokenizer),
            remove_columns=["messages"]
        )
        split_ds = dataset["train"].train_test_split(test_size=config.get("test_size", 0.1))
        
        # Prepare LoRA configuration and trainer
        lora_config = get_lora_config()
        trainer = create_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=split_ds["train"],
            eval_dataset=split_ds["test"],
            lora_config=lora_config,
            training_args=config["training_args"]
        )
        
        logger.info("Starting training...")
        trainer.train()
        trainer.save_model(config["output_dir"])
        logger.info("Training complete. Model saved to %s", config["output_dir"])
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a model using LoRA and function-calling data"
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()
    main(args.config)
