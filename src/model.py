# pip install bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from mlx_lm import generate, load

# model, tokenizer = load("google/gemma-2-9b")
model, tokenizer = load("mlx-community/gemma-2-9b-4bit")
#model, tokenizer = load("mlx-community/gemma-2-9b-8bit")

print(tokenizer.eos_token, tokenizer.bos_token)

# Generating without adding a prompt template manually
prompt = """
Provide a detailed explanation of why the sky is blue. Explain this to me as if I am a young child, but do not be afraid of using the real scientific terminiology or concepts. Make sure your explanation logically builds on itself.
""".strip()

prompt2 = "Recite the first 20 nuumbers of the fibonacci sequence"

prompt3 = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nContinue the fibonnaci sequence to the next 10 numbers.\n\n### Input:\n1, 1, 2, 3, 5, 8\n\n### Response:\n'

response = generate(
    model,
    tokenizer,
    prompt= tokenizer.bos_token + prompt3 + tokenizer.eos_token,
    verbose=True,  # Set to True to see the prompt and response
    temp=0.0,
    max_tokens=256,
)

print(f"Response: {response}")
