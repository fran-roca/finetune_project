from setuptools import setup, find_packages

setup(
    name="my_finetune_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "datasets",
        "peft",
        "trl",
        "pyyaml",
        "torch",
        "wandb",        # if needed
        "tensorboardX", # if needed
    ],
    entry_points={
        "console_scripts": [
            "train_model=scripts.train:main",
            "evaluate_model=scripts.evaluate:main",
            "push_model=scripts.push_to_hub:main",
        ],
    },
)