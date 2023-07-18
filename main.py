import torch
import clip
import coremltools as ct
import numpy as np
from PIL import Image

# device="cpu"
# model, preprocess = clip.load("ViT-B/16", device=device)
# text = clip.tokenize(["a diagram", "a dog", "a cat", "a bonsai tree", "a cherry tree outdoors"]).to(device)
# i = Image.open("/Users/ruijie.xiao/Downloads/media_sdk_ic_denoise@3x.jpg")
# image = preprocess(i).unsqueeze(0).to(device)
# traced = torch.jit.trace(model, (image, text))
#
# ct.convert(traced, inputs=[
#     ct.TensorType(name="image", shape=image.shape),
#     ct.TensorType(name="text", shape=text.shape, dtype=np.int64),
# ])

import torch
from PIL import Image
import open_clip

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2B-s34B-b88K', device=device)
# tokenizer = open_clip.get_tokenizer('ViT-B/16')
#
# image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
# text = tokenizer(["a diagram", "a dog", "a cat"])
#
# with torch.no_grad(), torch.cuda.amp.autocast():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#
#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#
# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


image = preprocess(Image.open("/Users/ruijie.xiao/Downloads/media_sdk_ic_denoise@3x.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "green", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

# traced = torch.jit.trace(model, (image, text))
traced = torch.jit.trace(model, image)

mlmodel = ct.convert(traced, inputs=[
    ct.TensorType(name="image", shape=image.shape),
    # ct.TensorType(name="text", shape=text.shape, dtype=np.int64),
])

mlmodel.save("newmodel.mlmodel")
