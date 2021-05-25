from typing import Dict

import torch
from torch import Tensor
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchaudio.models.wav2vec2.utils.import_huggingface import import_huggingface_model
from transformers import Wav2Vec2ForCTC

model = import_huggingface_model(Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h"))
model.eval()

# Remove weight normalization which is not supported by quantization.
model.encoder.transformer.pos_conv_embed.__prepare_scriptable__()

# Temporary wrapper to match the return type from
# https://github.com/pytorch/ios-demo-app/blob/master/SpeechRecognition/create_wav2vec2.py
# Once the resulting TorchScript object is confirmed to work, it is recommended to
# remove this wrapper and update the iOS code.
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform: Tensor) -> Dict[str, Tensor]:
        logit, _ = self.model(waveform)
        return {'logit': logit}


model = Wrapper(model)
quantized_model = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
scripted_model = torch.jit.script(quantized_model)
optimized_model = optimize_for_mobile(scripted_model)

# Sanity check
optimized_model(torch.randn(3, 1024))

optimized_model.save("wav2vec2.pt")
