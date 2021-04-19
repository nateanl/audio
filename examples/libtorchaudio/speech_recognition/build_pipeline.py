#!/usr/bin/evn python3
"""Build Speech Recognition pipeline based on fairseq's wav2vec2.0 and dump it to TorchScript file.

To use this script, you need `fairseq`.
"""
import os
import argparse
import logging

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import torchaudio
from torchaudio.models.wav2vec2.utils.import_fairseq import import_fairseq_model
import fairseq
import simple_ctc


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        '--model-file',
        required=True,
        help='Path to the input pretrained weight file.'
    )
    parser.add_argument(
        '--dict-dir',
        help=(
            'Path to the directory in which `dict.ltr.txt` file is found. '
            'Required only when the model is finetuned.'
        )
    )
    parser.add_argument(
        '--output-path',
        help='Path to the directory, where the TorchScript-ed pipelines are saved.',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help=(
            'When enabled, individual components are separately tested '
            'for the numerical compatibility and TorchScript compatibility.'
        )
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Apply quantization to model.'
    )
    parser.add_argument(
        '--optimize-for-mobile',
        action='store_true',
        help='Apply optmization for mobile.'
    )
    return parser.parse_args()


class Loader(torch.nn.Module):
    def forward(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, float(sample_rate), 16000.)
        return waveform


class Encoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(waveform)
        result = self.encoder(waveform, mask)['encoder_out'].transpose(1, 0)
        return result


class Decoder(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, emission: torch.Tensor) -> str:
        result = self.decoder.decode(emission)
        return ''.join(result.label_sequences[0][0]).replace('|', ' ')


def _load_fairseq_model(input_file, data_dir=None):
    overrides = {}
    if data_dir:
        overrides['data'] = data_dir

    model, args, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [input_file], arg_overrides=overrides
    )
    model = model[0]
    return model


def _get_decoder():
    labels = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "|",
        "E",
        "T",
        "A",
        "O",
        "N",
        "I",
        "H",
        "S",
        "R",
        "D",
        "L",
        "U",
        "M",
        "W",
        "C",
        "F",
        "G",
        "Y",
        "P",
        "B",
        "V",
        "K",
        "'",
        "X",
        "J",
        "Q",
        "Z",
    ]

    return Decoder(
        simple_ctc.BeamSearchDecoder(
            labels,
            cutoff_top_n=40,
            cutoff_prob=0.8,
            beam_size=100,
            num_processes=1,
            blank_id=0,
            is_nll=True,
        )
    )


def _quantize(model):
    custom_module_config = {
        'float_to_observed_custom_module_class': {
            torch.nn.MultiheadAttention: torch.nn.quantizable.MultiheadAttention
        },
        'observed_to_quantized_custom_module_class': {
            torch.nn.quantizable.MultiheadAttention: torch.nn.quantizable.MultiheadAttention
        }
    }
    model.qconfig = torch.quantization.get_default_qconfig(
        torch.backends.quantized.engine)
    model_prepared = torch.quantization.prepare(
        model, prepare_custom_config_dict=custom_module_config)
    model = torch.quantization.convert(
        model_prepared,
        convert_custom_config_dict=custom_module_config)
    print('Quantized:')
    print(model)
    return model


def _get_encoder(model_file, dict_dir, quantize, debug):
    original = _load_fairseq_model(model_file, dict_dir)
    model = import_fairseq_model(original, debug)
    print('Imported:')
    print(model)
    encoder = Encoder(model)
    if quantize:
        encoder = _quantize(encoder)
    return encoder


def _main():
    args = _parse_args()
    _init_logging(args.debug)
    loader = Loader()
    encoder = _get_encoder(args.model_file, args.dict_dir, args.quantize, args.debug)
    decoder = _get_decoder()

    torch.jit.script(loader).save(os.path.join(args.output_path, 'loader.zip'))
    torch.jit.script(decoder).save(os.path.join(args.output_path, 'decoder.zip'))
    scripted = torch.jit.script(encoder)
    if args.optimize_for_mobile:
        scripted = optimize_for_mobile(scripted)
    scripted.save(os.path.join(args.output_path, 'encoder.zip'))


def _init_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    format_ = (
        '%(message)s' if not debug else
        '%(asctime)s: %(levelname)7s: %(funcName)10s: %(message)s'
    )
    logging.basicConfig(level=level, format=format_)


if __name__ == '__main__':
    _main()
