# My Finetune Project

This project demonstrates how to fine-tune a language model using LoRA and function-calling data. The code is modular and production-ready, fully configurable to work with different models and datasets. It automatically detects if an NVIDIA CUDA GPU is available and uses it if possible.

## Project Structure

```
my_finetune_project/
├── .gitignore                # Git ignore patterns
├── README.md                 # This file
├── config/
│   └── default.yaml          # Model, dataset, and training hyperparameters
├── logs/                     # Log files (auto-created)
├── requirements.txt          # Python package dependencies
├── setup.py                  # Package setup and CLI entry points
├── src/
│   ├── __init__.py
│   ├── data/                 # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── preprocess.py     # Dataset preprocessing functions
│   ├── evaluation/           # Inference and evaluation routines
│   │   ├── __init__.py
│   │   └── evaluator.py      # Model evaluation logic
│   ├── models/               # Model & tokenizer factories and LoRA configuration
│   │   ├── __init__.py
│   │   ├── lora_config.py    # LoRA parameters configuration
│   │   ├── model_factory.py  # Model loading factory
│   │   └── tokenizer.py      # Tokenizer configuration
│   ├── training/             # Trainer factory and training routines
│   │   ├── __init__.py
│   │   └── trainer_factory.py # SFT trainer setup
│   └── utils/                # Config loader, logging, and device utilities
│       ├── __init__.py
│       ├── config_loader.py  # YAML configuration loader
│       ├── config_validator.py # Configuration validation
│       ├── device.py         # Device detection utility
│       └── logger.py         # Logging setup
├── scripts/                  # CLI scripts for train, evaluate, and push operations
│   ├── evaluate.py           # Model evaluation script
│   ├── push_to_hub.py        # HuggingFace Hub upload script
│   └── train.py              # Training script
└── tests/                    # Unit tests
    ├── __init__.py
    └── test_pipeline.py      # Test for data and tokenization pipeline
```

## Configure the Project

Edit the file `config/default.yaml` to set your model, dataset, and hyperparameters. Many training parameters have defaults that work for most cases—you need only override those that differ. For example, you can specify:

- **Model Section:**  
  Set the model name (and optionally the pad token, chat template, and special tokens) to switch to a different model.
- **Dataset Name:**  
  Change the dataset from `"Jofthomas/hermes-function-calling-thinking-V1"` to any other dataset.
- **Training Hyperparameters:**  
  Override parameters like batch size, learning rate, and number of epochs as needed.

## Features

- **Automatic Device Detection:** The code automatically detects NVIDIA CUDA GPUs and configures models appropriately
- **LoRA Fine-tuning:** Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA for efficient training
- **Configurable Chat Templates:** Support for custom chat templates to format conversational datasets
- **Modular Architecture:** Factories for models, tokenizers, and trainers allow easy customization
- **Comprehensive Logging:** Detailed logging to both console and files for debugging and monitoring
- **Error Handling:** Robust error handling and validation throughout the codebase

## Usage

### Train the Model

To train the model using your configuration:

```bash
python scripts/train.py --config config/default.yaml
```

The script loads your configuration, initializes the tokenizer and model (using defaults if necessary), preprocesses the dataset, and runs the training loop. The trained model is then saved to the directory specified by `output_dir` in the configuration.

### Evaluate the Model

To evaluate the trained model, run:

```bash
python scripts/evaluate.py --config config/default.yaml
```

You will be prompted to enter an input prompt. The script loads the model and tokenizer based on your configuration, runs inference, and prints the model's output.

### Push the Model (Optional)

If you want to push your model and tokenizer to the Hugging Face Hub, set `push_to_hub: true` in your configuration file. Then run:

```bash
python scripts/push_to_hub.py --config config/default.yaml
```

If the flag is set to false, the script logs that the model is stored locally.

## Testing

To run unit tests, use:

```bash
pytest tests/
```

## Dependencies

Install all required Python packages with:

```bash
pip install -r requirements.txt
```

Main dependencies:
- `transformers`: For model and tokenizer loading
- `peft`: For LoRA fine-tuning
- `trl`: For Supervised Fine-Tuning (SFT) trainer
- `datasets`: For dataset loading and preprocessing
- `torch`: PyTorch backend
- `pyyaml`: Configuration file parsing
- `wandb` (optional): For experiment tracking
- `tensorboardX` (optional): For TensorBoard integration

## Installation

You can also install the package with:

```bash
python setup.py install
```

This will create CLI entry points for training, evaluation, and pushing the model.

## Error Handling

The codebase includes comprehensive error handling:
- Configuration validation before training starts
- Graceful handling of model loading failures
- Detailed logging to help diagnose issues

## Additional Information

### Device Support
The code automatically detects if an NVIDIA CUDA GPU is available. If so, it uses the GPU (with an appropriate device map and bfloat16 conversion) for faster training and inference; otherwise, it falls back to CPU.

### Customization
The project uses factory patterns and dependency injection so you can easily switch models or datasets by simply updating the YAML configuration without modifying the code.

## License
[Specify your license here]
