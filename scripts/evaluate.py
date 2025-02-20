#!/usr/bin/env python
import argparse
import logging
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.models.tokenizer import TokenizerFactory
from src.models.model_factory import ModelFactory
from src.evaluation.evaluator import evaluate_model

def main(config_path: str) -> None:
    logger = setup_logger("evaluate", logging.INFO, "logs/evaluate.log")
    config = ConfigLoader.load(config_path)
    
    model_config = config["model"]
    tokenizer = TokenizerFactory.create(model_config)
    model = ModelFactory.create(model_config, tokenizer)
    
    prompt = input("Enter your prompt: ")
    try:
        output = evaluate_model(model, tokenizer, prompt)
        print("Model output:\n", output)
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        print("Evaluation failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned model"
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()
    main(args.config)
