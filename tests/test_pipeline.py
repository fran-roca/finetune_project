import pytest
from src.data.preprocess import preprocess_sample
from src.models.tokenizer import TokenizerFactory

@pytest.fixture
def sample_data():
    return {
        "messages": [
            {"role": "system", "content": "System information."},
            {"role": "user", "content": "User query content."}
        ]
    }

@pytest.fixture
def model_config():
    return {
        "name": "google/gemma-2-2b-it",
        "pad_token": "<pad>",
        "chat_template": (
            "{{ bos_token }}{% if messages[0]['role'] == 'system' %}"
            "{{ raise_exception('System role not supported') }}{% endif %}"
            "{% for message in messages %}"
            "{{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] | trim + '<end_of_turn><eos>\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}"
        ),
        "special_tokens": [
            "<pad>", "<eos>", "<tools>", "</tools>", "<think>",
            "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"
        ]
    }

@pytest.fixture
def tokenizer(model_config):
    return TokenizerFactory.create(model_config)

def test_preprocess_sample(sample_data, tokenizer):
    processed = preprocess_sample(sample_data, tokenizer)
    # Check that the output has a "text" key and that system info is merged
    assert "text" in processed
    assert "System information." in processed["text"]
    assert "User query content." in processed["text"]
