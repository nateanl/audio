import contextlib
import logging
import math
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter
from typing import Tuple, List

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.models.wav2vec2.components as components
from dataset import BucketizeBatchSampler, LibriSpeechFineTune
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class _Formatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    # https://stackoverflow.com/a/18462760
    pass


class TriStageLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warm up, hold, and decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        hold_updates: int,
        decay_updates: int,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.05,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.hold_updates = hold_updates
        self.decay_updates = decay_updates
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale

        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [
                base_lr * (self.init_lr_scale + self._step_count / self.warmup_updates * (1 - self.init_lr_scale))
                for base_lr in self.base_lrs
            ]
        elif self.warmup_updates < self._step_count <= (self.warmup_updates + self.hold_updates):
            return [base_lr for base_lr in self.base_lrs]
        elif self._step_count <= (self.warmup_updates + self.hold_updates + self.decay_updates):
            return [
                base_lr
                * math.exp(
                    math.log(self.final_lr_scale)
                    * (self._step_count - self.warmup_updates - self.hold_updates)
                    / self.decay_updates
                )
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr * self.final_lr_scale for base_lr in self.base_lrs]


Batch = Tuple[Tensor, int, str, int, int, int]


def _get_dict():
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    labels = bundle.get_labels()
    return {char: i for i, char in enumerate(labels)}


class CollateFnLibriSpeechFineTune:
    """The collate class for LibriSpeech or LibriSpeechFineTune dataset."""

    def __call__(self, batch: Batch) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            batch (Batch):
                The list of tuples that contains
                waveform, sample_rate, transcript, speaker_id, chapter_id, and utterance_id.

        Returns:
            (Tuple(Tensor, Tensor, Tensor, Tensor)):
                The Tensor of waveforms of dimension `[batch, time]`.
                The Tensor of labels of dimension `[batch, seq]`.
                The Tensor of audio lengths of dimension `[batch,]`.
                The Tensor of length lengths of dimension `[batch,]`.

        """
        audio_sizes = [sample[0].shape[1] for sample in batch]
        audio_size = max(audio_sizes)
        waveforms, labels, audio_lengths, label_lengths = [], [], [], []
        char_dict = _get_dict()
        for sample in batch:
            waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = sample
            label = torch.tensor([char_dict[e] for e in transcript.replace(" ", "|").upper()])
            audio_length = waveform.size(1)
            label_length = label.size(0)
            waveforms.append(waveform)
            audio_lengths.append(audio_length)
            label_lengths.append(label_length)
            labels.append(label)

        data = torch.zeros(len(batch), audio_size)
        for i in range(len(waveforms)):
            data[i][0 : waveforms[i].shape[1]] = waveforms[i]
        audio_lengths = torch.tensor(audio_lengths)
        label_lengths = torch.tensor(label_lengths)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
        return data, labels.int(), audio_lengths.int(), label_lengths.int()


def _get_lengths(files: List[str], path: str, ext_audio: str) -> List[int]:
    lengths = []
    for file_path in files:
        length = torchaudio.info(file_path).num_frames
        lengths.append(length)
    return lengths


def _get_lengths_librispeech(files: List[str], path: str, ext_audio: str) -> List[int]:
    lengths = []
    for file_path in files:
        speaker_id, chapter_id, utterance_id = file_path.split("-")
        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
        file_audio = fileid_audio + ext_audio
        file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)
        length = torchaudio.info(file_audio).num_frames
        lengths.append(length)
    return lengths


class HuBERTFineTuneModule(LightningModule):
    def __init__(
        self,
        *,
        model_name: str,
        aux_num_out: int,
        checkpoint: str,
        dataset: str,
        root_path: str,
        seconds_per_batch: float,
        learning_rate: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        fix_encoder_updates: int,
        warmup_updates: int,
        hold_updates: int,
        decay_updates: int,
    ):
        super().__init__()

        if model_name == "hubert_pretrain_base":
            self.model = torchaudio.models.hubert_pretrain_model(
                extractor_mode="group_norm",
                extractor_conv_layer_config=None,
                extractor_conv_bias=False,
                encoder_embed_dim=768,
                encoder_projection_dropout=0.0,
                encoder_pos_conv_kernel=128,
                encoder_pos_conv_groups=16,
                encoder_num_layers=12,
                encoder_num_heads=12,
                encoder_attention_dropout=0.0,
                encoder_ff_interm_features=3072,
                encoder_ff_interm_dropout=0.1,
                encoder_dropout=0.0,
                encoder_layer_norm_first=False,
                encoder_layer_drop=0.1,
                mask_prob=0.75,
                mask_selection="static",
                mask_other=0.0,
                mask_length=10,
                no_mask_overlap=False,
                mask_min_space=1,
                mask_channel_prob=0.5,
                mask_channel_selection="static",
                mask_channel_other=0.0,
                mask_channel_length=64,
                no_mask_channel_overlap=False,
                mask_channel_min_space=1,
                skip_masked=False,
                skip_nomask=False,
                num_classes=500,
                final_dim=256,
            )
        elif model_name == "hubert_large":
            self.model = torchaudio.models.hubert_large(aux_num_out=aux_num_out)
        elif model_name == "hubert_xlarge":
            self.model = torchaudio.models.hubert_xlarge(aux_num_out=aux_num_out)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        self.aux = torch.nn.Linear(768, 29)
        self._load_checkpoint(checkpoint)
        self.loss_fn = torch.nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
        self.optimizer = torch.optim.Adam(
            list(self.aux.parameters()) + list(self.model.wav2vec2.encoder.parameters()),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.fix_encoder_updates = fix_encoder_updates
        self.lr_scheduler = TriStageLRScheduler(self.optimizer, warmup_updates, hold_updates, decay_updates)
        self.dataset = dataset
        self.root_path = root_path
        self.seconds_per_batch = seconds_per_batch

    def _load_checkpoint(self, checkpoint):
        # load pretrain model
        state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
        state_dict = state_dict["state_dict"]
        s = {}
        for k in state_dict:
            if "wav2vec2" in k:
                s[k.replace("model.wav2vec2.", "")] = state_dict[k]
        self.model.wav2vec2.load_state_dict(s)

        # # this is for loading fairseq pretrain model
        # bundle = torchaudio.pipelines.HUBERT_BASE
        # m = bundle.get_model()
        # self.model.wav2vec2.load_state_dict(m.state_dict(), strict=True)

    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            return None
        waveforms, labels, audio_lengths, label_lengths = batch
        with torch.no_grad() if self.global_step <= self.fix_encoder_updates else contextlib.ExitStack():
            x, out_len = self.model.wav2vec2.feature_extractor(waveforms, audio_lengths)
            padding_mask = components._get_padding_mask(x, out_len)
            x, attention_mask = self.model.wav2vec2.encoder._preprocess(x, out_len)
            x, mask = self.model.mask_generator(x, padding_mask)
            x = self.model.wav2vec2.encoder.transformer(x, attention_mask=attention_mask)
        logits = self.aux(x)
        logits[padding_mask][..., 0] = 0
        logits[padding_mask][..., 1:] = float("-inf")
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)
        loss = self.loss_fn(
            log_probs,
            labels,
            out_len,
            label_lengths,
        )
        sample_size = logits.shape[0] * logits.shape[1] - torch.sum(padding_mask)
        loss = loss / sample_size * 10000
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return (
            [
                self.optimizer,
            ],
            [
                {"scheduler": self.lr_scheduler, "interval": "step"},
            ],
        )

    def training_step(self, batch: Batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def train_dataloader(self):
        dataset = LibriSpeechFineTune(self.root_path, "10h")
        lengths = _get_lengths(dataset._files, dataset._path, dataset._ext_audio)
        sampler = BucketizeBatchSampler(lengths, num_buckets=1000, max_token_count=3_200_000)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnLibriSpeechFineTune(),
            num_workers=10,
        )
        return dataloader

    def val_dataloader(self):
        dataset = torchaudio.datasets.LIBRISPEECH(self.root_path, "dev-other")
        lengths = _get_lengths_librispeech(dataset._walker, dataset._path, dataset._ext_audio)
        sampler = BucketizeBatchSampler(lengths, num_buckets=1000, max_token_count=3_200_000, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnLibriSpeechFineTune(),
            num_workers=10,
        )
        return dataloader


def run_train(args):
    checkpoint_dir = args.exp_dir / f"checkpoints_{args.dataset}_{args.model_name}"
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=True,
        verbose=True,
    )
    callbacks = [
        train_checkpoint,
    ]
    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_steps=args.max_updates,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        accelerator="gpu",
        strategy="ddp",
        replace_sampler_ddp=False,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
    )

    model = HuBERTFineTuneModule(
        model_name=args.model_name,
        aux_num_out=args.aux_num_out,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        root_path=args.root_path,
        seconds_per_batch=args.seconds_per_batch,
        learning_rate=args.learning_rate,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
        fix_encoder_updates=args.fix_encoder_updates,
        warmup_updates=args.warmup_updates,
        hold_updates=args.hold_updates,
        decay_updates=args.decay_updates,
    )
    trainer.fit(model)


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=_Formatter,
    )
    parser.add_argument(
        "--root-path",
        type=pathlib.Path,
        required=True,
        help="Path to the feature and label directories.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the feature and label directories.",
    )
    parser.add_argument(
        "--model-name",
        default="hubert_pretrain_base",
        choices=["hubert_pretrain_base", "hubert_pretrain_large", "hubert_pretrain_xlarge"],
        type=str,
        help="The HuBERT model to fine-tune.",
    )
    parser.add_argument(
        "--aux-num-out",
        default=29,
        type=int,
        help="The HuBERT model to train.",
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp_finetune_iter1"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--dataset",
        default="librispeech",
        choices=["librispeech", "librilight"],
        type=str,
        help="The dataset for training. (Default: 'librispeech')",
    )
    parser.add_argument(
        "--learning-rate",
        default=0.003,
        type=float,
    )
    parser.add_argument(
        "--betas",
        default=(0.9, 0.98),
        type=Tuple,
        help=" coefficients for computing running averages of gradient and its square (default: (0.9, 0.98))",
    )
    parser.add_argument(
        "--eps",
        default=1e-6,
        type=float,
        help="Epsilon value in Adam optimizer. (Default: 1e-6)",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.01,
        type=float,
        help="Weight decay (L2 penalty) (default: 0.01)",
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 8)",
    )
    parser.add_argument(
        "--fix-encoder-updates",
        default=10000,
        type=int,
        help="Number of steps for warm up the learning rate. (Default: 32000)",
    )
    parser.add_argument(
        "--warmup-updates",
        default=2000,
        type=int,
        help="Number of steps for warm up the learning rate. (Default: 32000)",
    )
    parser.add_argument(
        "--hold-updates",
        default=8000,
        type=int,
        help="Total number of training steps. (Default: 250000)",
    )
    parser.add_argument(
        "--decay-updates",
        default=10000,
        type=int,
        help="Total number of training steps. (Default: 250000)",
    )
    parser.add_argument(
        "--max-updates",
        default=25000,
        type=int,
        help="Total number of training steps. (Default: 250000)",
    )
    parser.add_argument(
        "--seconds-per-batch",
        default=200,
        type=float,
        help="Number of seconds of audio in a mini-batch. (Default: 87.5)",
    )
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def _init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = _parse_args()
    _init_logger(args.debug)
    run_train(args)
    # _debug(args)


if __name__ == "__main__":
    cli_main()
