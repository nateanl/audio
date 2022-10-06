import os
from pathlib import Path
from typing import Tuple, Union

import torch
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform


SAMPLE_RATE = 16000


def _load_transcripts(file: Path, subset: str):
    transcripts = {}
    with open(file, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            index = line[0]
            trans = " ".join(line[1:])
            if subset in index:
                transcripts[index] = trans
    return transcripts


class Snips(Dataset):
    """*Snips* :cite:`coucke2018snips` dataset.

    Args:
        root (str or Path): Root directory where the dataset's top level directory is found
        subset (str): Subset of the dataset to use. Options: [``"train"``, ``"valid"``, ``"test"``].
    """

    _ext_audio = ".wav"
    _trans_file = "all.iob.snips.txt"

    def __init__(
        self,
        root: Union[str, Path],
        subset: str,
    ) -> None:
        if subset not in ["train", "valid", "test"]:
            raise ValueError('`subset` must be one of ["train", "valid", "test"]')

        root = Path(root)
        self._path = root / "SNIPS"
        self.audio_path = self._path / subset

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found.")

        self.audio_paths = self.audio_path.glob(f"*{self._ext_audio}")
        self.data = []
        for audio_path in self.audio_paths:
            audio_stem = str(audio_path.stem)
            self.data.append(audio_stem)
        transcript_path = self._path / self._trans_file
        self.transcripts = _load_transcripts(transcript_path, subset)

    def get_metadata(self, n: int) -> Tuple[str, int, str]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            str:
                Path to audio
            int:
                Sample rate
            str:
                Transcription of audio
        """
        audio_path = self.data[n]
        relpath = os.path.relpath(audio_path, self._path)
        file_name = audio_path.with_suffix("").name
        transcripte = self.transcripts[file_name]
        return relpath, SAMPLE_RATE, transcripte

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcription of audio
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._path, metadata[0], metadata[1])
        return (waveform,) + metadata[1:]

    def __len__(self) -> int:
        return len(self.data)
