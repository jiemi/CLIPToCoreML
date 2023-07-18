# -*- coding: UTF-8 -*-

import coremltools as ct
import numpy as np
from PIL import Image
import torch
import clip
import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", return_dict=False)


# model_pt = clip.load('/Users/ruijie.xiao/Downloads/ViT-B-16.pt')

model_pt = model.text_model
model.eval()
model_pt.eval()

# Trace
# wrapped_model -> wrapped CLIPModel so that forward() function returns get_image_features()
example_input = torch.rand(1, 3, 224, 224)

text = "a painting of a corgi wearing a royal cape and crown"
# Since we're tracing, should probably pad the text so that different length inputs are pre-processed to a single input shape
preprocessed_text = processor(text=text, padding='max_length', return_tensors='pt').input_ids

model_traced = torch.jit.trace(model_pt, preprocessed_text,strict=False)

# Convert traced model to CoreML
model_coreml = ct.convert(
    model_traced,
    inputs=[ct.TensorType(name="input_image", shape=preprocessed_text.shape)]
)

model_coreml.save("mlmodel_vision.mlmodel")

# Inference through all 3 models. Convert to numpy for easier comparison
# image = ... # Load real image from path
# processed_image = processor(text= None, images=[image], return_tensors="pt", padding=True)
#
# res_pt = model_pt.get_image_features(processed_image).numpy()
# res_traced = model_traced(processed_image).numpy()
# res_coreml = model_coreml.predict({'input_image': processed_image.numpy()})['output_name']
#
#
# # Compare outputs
# print(np.array_equal(res_pt, res_traced)) # True -> standard and traced model produce the same results
# print(np.array_equal(res_pt, res_coreml)) # False -> different output
#
# # How close are the outputs?
# print(np.allclose(res_pt, res_coreml, atol=1e-5)) # False
# print(np.allclose(res_pt, res_coreml, atol=1e-2)) # False
# print(np.allclose(res_pt, res_coreml, atol=1e-1)) # True -> Results differ ~0.1