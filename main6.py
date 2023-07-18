import coremltools as ct
import clip
from PIL import Image
import torch
import numpy as np

#https://github.com/apple/coremltools/issues/1868

from torch import nn
import torch
class TextTransformer(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # self.param = torch.nn.Parameter(torch.tensor(2.))
        self.clip_model = clip_model

    def forward(self, x: torch.Tensor):

        return self.clip_model.encode_text(x)

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

    text = clip.tokenize(["a diagram", "green", "a cat"]).to("cpu")

    text_model = TextTransformer(model)
    input = ct.ImageType(
        name="text",
        shape=text.shape,
        scale=scale,
        bias=bias,
    )

    tr_model = torch.jit.trace(text_model, text)

    ml_model = ct.convert(
        source="auto",
        model=tr_model,
        inputs=[
            ct.TensorType(name="text",shape=text.shape, dtype=np.int64)
        ],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
    )

    ml_model.save("text_encoder.mlpackage")


if __name__ == "__main__":
    main()