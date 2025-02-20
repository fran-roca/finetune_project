from typing import Dict, Any, List

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validates the configuration dictionary and returns a list of validation errors.

    Args:
        config (Dict[str, Any]): Configuration dictionary to validate

    Returns:
        List[str]: List of validation error messages; empty if no errors.
    """
    errors = []
    
    # Check required top-level keys
    required_keys = ["model", "dataset_name", "output_dir", "training_args"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required configuration key: {key}")
    
    # Check model configuration
    if "model" in config:
        model_config = config["model"]
        if "name" not in model_config:
            errors.append("Model configuration missing 'name' property")
    
    # Check training arguments
    if "training_args" in config:
        training_args = config["training_args"]
        required_training_args = ["output_dir", "per_device_train_batch_size", "learning_rate"]
        for arg in required_training_args:
            if arg not in training_args:
                errors.append(f"Training arguments missing required parameter: {arg}")
    
    return errors
