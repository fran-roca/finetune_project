#!/usr/bin/env python
import argparse
from src.utils.config_loader import ConfigLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.logger import setup_logger

logger = setup_logger("push_to_hub")

def main(config_path: str) -> None:
    config = ConfigLoader.load(config_path)
    
    # Check if pushing to the hub is enabled; if not, exit early.
    if not config.get("push_to_hub", False):
        logger.info("Push to hub is disabled in the configuration. Model is saved locally at: %s", config["output_dir"])
        return

    username = config.get("username", "your_username")
    output_dir = config["output_dir"]
    
    # Load tokenizer and model from output_dir
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    
    # Ensure tokenizer is configured correctly
    tokenizer.eos_token = "<eos>"
    
    logger.info("Pushing model to the hub under %s/%s", username, output_dir)
    model.push_to_hub(f"{username}/{output_dir}")
    tokenizer.push_to_hub(f"{username}/{output_dir}", token=True)
    logger.info("Push complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push the trained model and tokenizer to the Hugging Face Hub if enabled in config."
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()
    main(args.config)
