import coremltools as ct
import clip
from PIL import Image
import torch
import numpy as np

#https://github.com/apple/coremltools/issues/1868

def main():
    scale = 1 / (0.2685697 * 255.0)
    bias = [
        -0.48145466 / (0.26862954),
        -0.4578275 / (0.26130258),
        -0.40821073 / (0.27577711),
    ]

    model, proc = clip.load("ViT-B/32", device="cpu")

    img = Image.new(mode="RGB", size=(1024, 1024))
    img = proc(img).unsqueeze(0).to(device="cpu")

    input = ct.ImageType(
        name="image",
        shape=img.shape,
        scale=scale,
        bias=bias,
    )

    tr_model = torch.jit.trace(model.visual, img)

    ml_model = ct.convert(1,
        tr_model,
        inputs=[input],
        outputs=[ct.TensorType(name="embedding")]
    )

    ml_model.save("image_encoder.mlpackage")


if __name__ == "__main__":
    main()