import transformers
import torch
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

login(token=TOKEN)

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
res = pipeline("Hey how are you doing today?")

print(f"{res=}")
