# MLX Dev Server

[Installation](#installation) | [Usage](#usage) | [Examples](#examples)

A simple solution to run LLMs locally on Macs with Apple Silicon. Optimized for code completion tasks with DeepSeek, Qwen and other models.

<img width="1046" alt="Screenshot" src="https://github.com/user-attachments/assets/0c2bdec7-1bfe-4a3b-9cfb-067cb34c036f" />

## Features

- 🚀 **Fast**: uses [Apple MLX](https://github.com/ml-explore/mlx) to run models on GPU using unified memory
- 💪 **Efficient**: cancels generation when client disconnects (see [Motivation](#motivation) on why it is important for code completion)
- 🧩 **Compatible**: provides OpenAI-like API to easily integrate with existing applications (see [Examples](#examples))
- 💾 **Memory Efficient**: unloads models when they are not used
- 🔗 **Reliable**: test coverage is 97%

## Motivation

While [Ollama](https://github.com/ollama/ollama) is effective for many tasks, it can be less responsive for code completion due to its handling of prompt processing.

Code completion requires quick processing of large inputs (1k+ tokens) and short output generation (<100 tokens typically). And most completions are cancelled because developers often pause for a moment and continue typing, discarding the completion. Ollama processes the entire prompt before cancellation, leading to potential delays.

MLX Dev Server addresses this by cancelling both prompt processing and generation when the client disconnects, ensuring consistent and responsive code completion.

## Installation

```bash
pip install mlx-dev-server
```

## Usage

Simply run `mlx_dev_server`.

Available command line arguments:


- `-p, --port`: Port to listen on (default is `8080`)
- `-k, --keep-alive`: Time in seconds to keep models loaded in memory (default is `300`)
- `-m, --max-loaded-models`: Maximum number of models to keep loaded (default is `2`)
- `--host`: Host to listen on (default is `localhost`)
- `--max-tokens`: Maximum tokens to generate if not specified (default is `4096`)
- `--max-kv-size`: Maximum size of the key-value cache (default is `4096`)
- `--kv-bits`: Bits for key-value cache quantization (default is `8`)
- `--prefill-step-size`: Step size for prompt processing (default is `128`)

## Examples

### VSCode

Install [llm-vscode](https://marketplace.visualstudio.com/items?itemName=HuggingFace.huggingface-vscode) extension. Then add the following to `settings.json`:

```json
{
    "llm.backend": "openai",
    "llm.url": "http://localhost:8080",
    "llm.modelId": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
    "llm.configTemplate": "Custom",
    "llm.requestBody": {
        "parameters": {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 60
        }
    },
    "llm.fillInTheMiddle.prefix": "<｜fim▁begin｜>",
    "llm.fillInTheMiddle.middle": "<｜fim▁end｜>",
    "llm.fillInTheMiddle.suffix": "<｜fim▁hole｜>",
    "llm.tokenizer": {
        "repository": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
    },
    "llm.contextWindow": 1024
}
```

> [!NOTE]
> This configuration limits the number of generated tokens to 60.
> This is to speed up the response of the model if it decides to generate a multi-line code snippet.

### Neovim

Add the following spec to [lazy.nvim](https://github.com/folke/lazy.nvim) configuration to enable [llm.nvim](https://github.com/huggingface/llm.nvim) plugin:
```lua
{
  'huggingface/llm.nvim',
  opts = {
    backend = 'openai',
    url = 'http://localhost:8080',
    model = 'mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx',
    request_body = {
      temperature = 0.2,
      top_p = 0.95,
      max_tokens = 60
    },
    fim = {
      prefix = '<｜fim▁begin｜>',
      middle = '<｜fim▁end｜>',
      suffix = '<｜fim▁hole｜>'
    },
    tokenizer = {
      repository = 'mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx'
    },
    context_window = 1024
  }
}
```

### OpenAI Python API library

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8080/v1',
    api_key='mlx-dev-server',  # not needed but required
)

response = client.chat.completions.create(
    model='mlx-community/Mistral-Nemo-Instruct-2407-8bit',
    messages=[{
        'role': 'user',
        'content': 'say hello',
    }],
)
print(response.choices[0].message.content)
```
