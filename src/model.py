# pip install bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from mlx_lm import generate, load

model, tokenizer = load("google/gemma-7b-it")
