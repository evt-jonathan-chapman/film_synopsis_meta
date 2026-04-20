MODELS = {
    # OpenAI API models (via LiteLLM)
    'gpt-4.1-nano': {
        'provider': 'openai',
        'model': 'gpt-4.1-nano',
    },
    'gpt-4o-mini': {
        'provider': 'openai',
        'model': 'gpt-4o-mini',
    },
    'gpt-5.4-nano': {
        'provider': 'openai',
        'model': 'gpt-5.4-nano',
        # Fallback pricing (USD per 1M tokens) used when LiteLLM doesn't know the model yet
        'cost_per_1m_input': 0.10,
        'cost_per_1m_output': 0.40,
    },

    # Local llama-cpp models
    # phi-3.5: disqualified — parse failures + hallucinations (2026-04-10 comparison)
    # 'phi-3.5': {
    #     'provider': 'llama_cpp',
    #     'model': 'openai/phi-3.5',
    #     'repo_id': 'bartowski/Phi-3.5-mini-instruct-gguf',
    #     'filename': 'Phi-3.5-mini-instruct-Q5_K_M.gguf',
    #     'n_gpu_layers': -1,
    #     'main_gpu': 0,
    # },
    'llama-3.1': {
        'provider': 'llama_cpp',
        'model': 'openai/llama-3.1',
        'repo_id': 'unsloth/Llama-3.1-8B-Instruct-GGUF',
        'filename': 'Llama-3.1-8B-Instruct-Q4_K_M.gguf',
        'n_gpu_layers': 24,
        'main_gpu': 0,
    },
}

# Production defaults: smallest capable OpenAI model first, larger mini as fallback.
DEFAULT_MODEL = 'gpt-4.1-nano'
DEFAULT_FALLBACKS = [{'model': 'gpt-4o-mini'}]
