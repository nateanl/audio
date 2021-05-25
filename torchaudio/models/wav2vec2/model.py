from typing import Optional, Tuple, List

from torch import Tensor
from torch.nn import Module

from . import components


class Wav2Vec2Model(Module):
    def __init__(
            self,
            feature_extractor: Module,
            encoder: Module,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder

    def extract_feature(
            self,
            waveforms: Tensor,
            lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self.feature_extractor(waveforms, lengths)

    def forward(
            self,
            waveforms: Tensor,
            lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute the sequence of probability distribution over labels.

        

        """
        x, lengths = self.feature_extractor(waveforms, lengths)
        return self.encoder(x, lengths), lengths


def _get_model(
        extractor_mode: str,
        extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
        extractor_conv_bias: bool,
        encoder_embed_dim: int,
        encoder_projection_dropout: float,
        encoder_pos_conv_kernel: int,
        encoder_pos_conv_groups: int,
        encoder_num_layers: int,
        encoder_num_heads: int,
        encoder_attention_dropout: float,
        encoder_ff_interm_features: int,
        encoder_ff_interm_dropout: float,
        encoder_dropout: float,
        encoder_layer_norm_first: bool,
        encoder_layer_drop: float,
        encoder_num_out: int,
) -> Wav2Vec2Model:
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    feature_extractor = components._get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias)
    encoder = components._get_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        num_heads=encoder_num_heads,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
        num_out=encoder_num_out,
    )
    return Wav2Vec2Model(feature_extractor, encoder)


def wav2vec2_base(num_out: int) -> Wav2Vec2Model:
    """Build wav2vec2.0 model with "Base" configuration.
    """
    return _get_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_attention_dropout=0.1,
        encoder_ff_interm_features=3072,
        encoder_ff_interm_dropout=0.1,
        encoder_dropout=0.1,
        encoder_layer_norm_first=True,
        encoder_layer_drop=0.1,
        encoder_num_out=num_out,
    )


def wav2vec2_large(num_out: int) -> Wav2Vec2Model:
    """Build wav2vec2.0 model with "Large" configuration.
    """
    return _get_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=0.0,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=0.1,
        encoder_dropout=0.0,
        encoder_layer_norm_first=False,
        encoder_layer_drop=0.1,
        encoder_num_out=num_out,
    )
