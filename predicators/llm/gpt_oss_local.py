# from transformers import pipeline
# import torch

# generator = pipeline(
#     "text-generation",
#     model="openai/gpt-oss-20b",
#     torch_dtype="auto",
#     device_map="auto",
# )

# messages = [
#     {"role": "user", "content": "Explain what MXFP4 quantization is."},
# ]

# result = generator(
#     messages,
#     max_new_tokens=5000,
#     temperature=1.0,
# )

# print(result[0]["generated_text"])

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)