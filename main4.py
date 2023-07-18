from transformers import AutoTokenizer, CLIPTextModel
import torch
import coremltools as ct
import numpy as np
import coremltools as ct

model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

traced_model = torch.jit.trace(model, [input_ids, attention_mask], strict=False)

model = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(shape=input_ids.shape),
        ct.TensorType(shape=attention_mask.shape)])

model.save("CLIPTextModel.mlpackage")