from llama_cpp import Llama

from models import MODELS

model = MODELS.get('llama-3.1')

llm = Llama.from_pretrained(
    repo_id=model.get('repo_id'),
    filename=model.get('filename'),
    n_ctx=2048,
    n_threads=8,
    verbose=True,
    seed=42,
    n_gpu_layers=model.get('n_gpu_layers', 0),
    main_gpu=1,
    n_batch=256,
)

llm(
    "Test GPU usage.",
    max_tokens=16,
)

llm.close()
