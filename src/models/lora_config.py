from peft import LoraConfig, TaskType

def get_lora_config() -> LoraConfig:
    """
    Returns a LoraConfig with predefined parameters.

    Returns:
        LoraConfig: The LoRA configuration.
    """
    return LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "gate_proj", "q_proj", "lm_head", "o_proj", "k_proj",
            "embed_tokens", "down_proj", "up_proj", "v_proj"
        ],
        task_type=TaskType.CAUSAL_LM
    )
