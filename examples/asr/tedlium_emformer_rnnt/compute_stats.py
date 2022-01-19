import json
import torch
import torchaudio


device = torch.device("cuda")
_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80, hop_length=160).to(device)
dataset = torchaudio.datasets.TEDLIUM("/fsx/users/zni/datasets/", release="release3", subset="train")

stats = None
length = 0.
for i, data in enumerate(dataset):
    waveform, sample_rate, transcript, talk_id, speaker_id, identifier = data
    waveform = waveform.to(device)
    mel_specgram = _spectrogram_transform(waveform)
    if stats is not None:
        stats += mel_specgram.cpu().sum(dim=-1)
    else:
        stats = mel_specgram.cpu().sum(dim=-1)
    length += mel_specgram.size(-1)

mean = stats / length

print(mean.shape)
std = None
length = 0.
for i, data in enumerate(dataset):
    waveform, sample_rate, transcript, talk_id, speaker_id, identifier = data
    waveform = waveform.to(device)
    mel_specgram = _spectrogram_transform(waveform)
    if std is not None:
        std += torch.sum((mel_specgram.cpu() - mean.unsqueeze(-1)) ** 2, dim=-1)
    else:
        std = torch.sum((mel_specgram.cpu() - mean.unsqueeze(-1)) ** 2, dim=-1)
    length += mel_specgram.size(-1)


invstd = 1 / torch.sqrt(std / length)

print(invstd.shape)
stats_dict = {
    "mean": mean.tolist(),
    "invstddev": invstd.tolist(),
}

with open("global_stats.json", "w") as f:
    json.dump(stats_dict, f, indent=2)