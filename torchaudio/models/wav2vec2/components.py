from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

# https://github.com/pytorch/fairseq/blob/b0ae834d528a4a466202107a22356aed71bb6161/fairseq/modules/gelu.py#L24
def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


# https://github.com/pytorch/fairseq/blob/b0ae834d528a4a466202107a22356aed71bb6161/fairseq/modules/fp32_group_norm.py#L13
# This is not TS-compatible
# TODO: check what happens in float16. Can we use the retular GroupNorm??
class Fp32GroupNorm(torch.nn.GroupNorm):
    def forward(self, input: Tensor) -> Tensor:
        # It's practically same as `super().forward(...)` but TS does not support `super()`.
        return torch.nn.functional.group_norm(
            input.float(), self.num_groups, self.weight, self.bias, self.eps).type_as(input)


# https://github.com/pytorch/fairseq/blob/b0ae834d528a4a466202107a22356aed71bb6161/fairseq/modules/layer_norm.py#L38
# This is not TS-compatible
# TODO: check what happens in float16. Can we use the regular LayerNorm??
class Fp32LayerNorm(torch.nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        # It's practically same as `super().forward(...)` but TS does not support `super()`.
        return torch.nn.functional.layer_norm(
            input.float(), self.normalized_shape, self.weight, self.bias, self.eps).type_as(input)


# https://github.com/pytorch/fairseq/blob/b0ae834d528a4a466202107a22356aed71bb6161/fairseq/modules/same_pad.py#L10
class SamePad(Module):
    def __init__(self, kernel_size: int, causal: bool = False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: Tensor) -> Tensor:
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


# https://github.com/pytorch/fairseq/blob/b0ae834d528a4a466202107a22356aed71bb6161/fairseq/modules/transpose_last.py#L12-L20
class TransposeLast(Module):
    def __init__(self, deconstruct_idx: Optional[int] = None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x: Tensor) -> Tensor:
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


# https://github.com/pytorch/fairseq/blob/b0ae834d528a4a466202107a22356aed71bb6161/fairseq/models/wav2vec/wav2vec2.py#L665
class ConvFeatureExtractionModel(Module):
    def __init__(self, conv_layers: Module):
        super().__init__()
        self.conv_layers = conv_layers

    def forward(self, x: Tensor) -> Tensor:
        # B x T -> B x C x T
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


# https://github.com/pytorch/fairseq/blob/b0ae834d528a4a466202107a22356aed71bb6161/fairseq/models/wav2vec/wav2vec2.py#L836
class TransformerSentenceEncoderLayer(Module):
    def __init__(
            self,
            self_attn: Module,
            dropout1: Module,
            dropout2: Module,
            dropout3: Module,
            self_attn_layer_norm: Module,
            fc1: Module,
            fc2: Module,
            final_layer_norm: Module,
            layer_norm_first: bool = False,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.self_attn_layer_norm = self_attn_layer_norm
        self.fc1 = fc1
        self.fc2 = fc2
        self.final_layer_norm = final_layer_norm
        self.layer_norm_first = layer_norm_first

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: Optional[Tensor] = None,
            self_attn_padding_mask: Optional[Tensor] = None,
            need_weights: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        # This is the stripped down version of
        # https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/models/wav2vec/wav2vec2.py#L883
        # with the following assumption
        # - self_attn_mask: torch.Tensor = None,
        # - self_attn_padding_mask: torch.Tensor = None,
        # - need_weights: bool = False,
        # - att_args=None,
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = gelu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = gelu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)
        return x, attn


# https://github.com/pytorch/fairseq/blob/b0ae834d528a4a466202107a22356aed71bb6161/fairseq/models/wav2vec/wav2vec2.py#L746
class TransformerEncoder(Module):
    def __init__(
            self,
            pos_conv: Module,
            layers: Module,
            layer_norm: Module,
            dropout: float = 0.1,
            layer_norm_first: bool = False,
    ):
        super().__init__()
        self.pos_conv = pos_conv
        self.layers = layers
        self.layer_norm = layer_norm

        self.dropout = dropout
        self.layer_norm_first = layer_norm_first

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x: Tensor, padding_mask: Optional[Tensor] = None):
        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Note: We do not implemente layer drop
        # https://github.com/pytorch/fairseq/blob/b0ae834d528a4a466202107a22356aed71bb6161/fairseq/models/wav2vec/wav2vec2.py#L815-L820
        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


# https://github.com/pytorch/fairseq/blob/b0ae834d528a4a466202107a22356aed71bb6161/fairseq/models/wav2vec/wav2vec2.py#L222
class Wav2Vec2Model(Module):
    def __init__(
            self,
            feature_extractor,
            post_extract_proj,
            dropout_input,
            dropout_features,
            encoder,
            layer_norm,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.post_extract_proj = post_extract_proj
        self.dropout_input = dropout_input
        self.dropout_features = dropout_features
        self.encoder = encoder
        self.layer_norm = layer_norm

    @torch.jit.export
    def extract_features(
            self,
            source: Tensor,
            padding_mask: Optional[Tensor] = None,
    ):
        # Based off of
        # https://github.com/pytorch/fairseq/blob/b0ae834d528a4a466202107a22356aed71bb6161/fairseq/models/wav2vec/wav2vec2.py#L474-L478
        #
        # with the following values
        #
        # mask = False
        # features_only = True
        # mask_indices = None
        # mask_channel_indices = None
        # padding_count = None

        features = self.feature_extractor(source)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[(torch.arange(padding_mask.shape[0], device=padding_mask.device), output_lengths - 1)] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).to(torch.bool)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        x = features
        x = self.encoder(x, padding_mask=padding_mask)
        return x, padding_mask

    @torch.jit.export
    def _get_feat_extract_output_lengths(self, input_lengths: Tensor):
        for layers in self.feature_extractor.conv_layers:
            kernel_size = layers[0].kernel_size[0]
            stride = layers[0].stride[0]
            input_lengths = torch.floor((input_lengths - kernel_size) / stride + 1)
        return input_lengths.to(torch.long)


# https://github.com/pytorch/fairseq/blob/b0ae834d528a4a466202107a22356aed71bb6161/fairseq/models/wav2vec/wav2vec2_asr.py#L290-L405
class Wav2VecEncoder(Module):
    def __init__(
            self,
            w2v_model: Module,
            final_dropout: Module,
            proj: Module,
    ):
        super().__init__()
        self.w2v_model = w2v_model
        self.final_dropout = final_dropout
        self.proj = proj

    def forward(
            self,
            source: Tensor,
            padding_mask: Optional[Tensor] = None,
            tbc: bool = True,
    ):
        x, padding_mask = self.w2v_model.extract_features(source, padding_mask)
        assert padding_mask is not None  # Trick to make TorchScript compiler assume padding_mask is not Optional
        if tbc:
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
        x = self.final_dropout(x)
        x = self.proj(x)
        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask.transpose(0, 1),  # T x B
            "padding_mask": padding_mask,
        }


# https://github.com/pytorch/fairseq/blob/b0ae834d528a4a466202107a22356aed71bb6161/fairseq/models/wav2vec/wav2vec2_asr.py#L153-L191
class Wav2VecCtc(Module):
    def __init__(self, w2v_encoder):
        super().__init__()
        self.w2v_encoder = w2v_encoder

    def forward(
            self,
            source: Tensor,
            padding_mask: Optional[Tensor] = None,
    ):
        return self.w2v_encoder(source, padding_mask)
