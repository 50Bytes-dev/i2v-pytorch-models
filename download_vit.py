import os
from transformers import ViTModel, ViTImageProcessor

# https://huggingface.co/SoBytes/vit-sadovod
MODEL_NAME = "SoBytes/vit-sadovod"

hf_home = os.getenv("HF_HOME", None)
if hf_home:
    os.makedirs(hf_home, exist_ok=True)

model = ViTModel.from_pretrained(MODEL_NAME, cache_dir=hf_home)
processor = ViTImageProcessor.from_pretrained(MODEL_NAME, cache_dir=hf_home)

print(f"Model and processor for {MODEL_NAME} downloaded successfully.")
