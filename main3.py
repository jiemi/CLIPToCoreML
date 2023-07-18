import torch
import numpy as np
import coremltools as ct

from transformers.models.clip.processing_clip import CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPModel


# Combine CLIP Text Model and Text Projection for exporting as one
# // Disabled text_projection as its not needed to reproduce error
class CLIPTextEmbedder(torch.nn.Module):
    def __init__(self, text_model):
        super(CLIPTextEmbedder, self).__init__()
        self.text_model = text_model
        # self.text_projection = text_projection

    def forward(self, input_ids):
        pooled_out = self.text_model(input_ids)[1]
        # embedded_out = self.text_projection(pooled_out)
        return pooled_out  # embedded_out



# Load the pretrained CLIP Model from Huggingface Transformers
pretrained_identifier = 'openai/clip-vit-base-patch16'
processor = CLIPProcessor.from_pretrained(pretrained_identifier)
# return_dict=False so that it returns tuples... needed for tracing
model = CLIPModel.from_pretrained(pretrained_identifier, return_dict=False)

# Isolate text embedding into separate models for export
text_for_export = CLIPTextEmbedder(model.text_model)  # , model.text_projection)

# Pre-processing text for trace
text = "a painting of a corgi wearing a royal cape and crown"

from coremltools.converters.mil.mil import types
# input1 = ct.TensorType(name='input_ids', shape=input_ids.size(), dtype=types.int64)

# Since we're tracing, should probably pad the text so that different length inputs are pre-processed to a single input shape
preprocessed_text = processor(text=text, padding='max_length', return_tensors='pt').input_ids

with torch.no_grad():
    traced_model = torch.jit.trace(text_for_export, preprocessed_text,strict=False)
    text_outputs = traced_model(preprocessed_text)
    ml_model = ct.convert(
        model=traced_model,
        inputs=[
            ct.TensorType(shape=preprocessed_text.shape, dtype=np.int64)
        ]
    )

# Export to CLIPTextEmbedder[openai_clip-vit-base-patch32].mlmodel
out_filename = f'{type(text_for_export).__name__}[{pretrained_identifier.replace("/", "_")}].mlmodel'
ml_model.save(out_filename)