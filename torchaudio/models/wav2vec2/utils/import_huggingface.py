"""Import Hugging Face transformers's wav2vec2.0 pretrained weights to torchaudios's format.

For this module to work, you need `transformers`.
"""

import logging
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import Module

from torchaudio._internal import module_utils as _mod_utils

_LG = logging.getLogger(__name__)


# NOTE: fairseq's original ConvLayerBlock had dropout (but with p=0.0)
class ConvLayer(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            bias: bool,
            norm_type: Optional[str],
    ):
        super().__init__()
        if norm_type is not None and norm_type not in ['group', 'layer']:
            raise ValueError('"norm_type" must be either None, "layer" or "group".')

        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

        if norm_type is None:
            self.layer_norm = None
        elif norm_type == 'group':
            self.layer_norm = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True
            )
        elif norm_type == 'layer':
            self.layer_norm = nn.LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True
            )

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            if self.norm_type == "group":
                x = self.layer_norm(x)
            elif self.norm_type == "layer":
                x = x.transpose(-2, -1)
                x = self.layer_norm(x)
                x = x.transpose(-2, -1)
        x = torch.nn.functional.gelu(x)
        return x


def _import_conv_layer(module: Module, norm_type: Optional[str], debug: bool):
    assert module.activation is nn.functional.gelu
    mod = ConvLayer(
        in_channels=module.conv.in_channels,
        out_channels=module.conv.out_channels,
        kernel_size=module.conv.kernel_size[0],
        stride=module.conv.stride[0],
        bias=norm_type == "layer",
        norm_type=norm_type,
    )
    mod.load_state_dict(module.state_dict())
    if debug:
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)

        input = torch.randn(3, module.conv.in_channels, 10)
        ref = original(input)
        hyp = imported(input)
        hyp_jit = scripted(input)
        torch.testing.assert_allclose(ref, hyp)
        torch.testing.assert_allclose(ref, hyp_jit)
    return mod


class FeatureExtractor(Module):
    def __init__(self, conv_layers: Module):
        super().__init__()
        self.conv_layers = conv_layers

    def forward(self, x):
        """
        Expected input shape: (batch, time)
        """
        x = x[:, None]  # (batch, channel == 1, time)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x

    def get_output_length(self, input_length):
        output_length = input_length
        for l in self.conv_layers:
            output_length = (input_length - l.kernel_size) // l.stride + 1
        return output_length


def _import_feature_extractor(module: Module, debug: bool, indent: int):
    mod = FeatureExtractor(conv_layers=_import(module.conv_layers, debug, indent))
    if debug:
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)

        input = torch.randn(3, 1024)
        ref = original(input)
        hyp = imported(input)
        hyp_jit = scripted(input)
        torch.testing.assert_allclose(ref, hyp)
        torch.testing.assert_allclose(ref, hyp_jit)
    return mod


class FeatureProjection(Module):
    def __init__(
            self,
            normalized_shape,
            layer_norm_eps: float,
            in_features: int,
            out_features: int,
            dropout: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=layer_norm_eps)
        self.projection = nn.Linear(in_features, out_features,)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


def _import_feature_projection(module: Module, debug: bool):
    mod = FeatureProjection(
        normalized_shape=module.layer_norm.normalized_shape,
        layer_norm_eps=module.layer_norm.eps,
        in_features=module.projection.in_features,
        out_features=module.projection.out_features,
        dropout=module.dropout.p,
    )
    mod.load_state_dict(module.state_dict())
    if debug:
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)

        input = torch.randn(3, module.projection.in_features)
        ref = original(input)
        hyp = imported(input)
        hyp_jit = scripted(input)
        torch.testing.assert_allclose(ref, hyp)
        torch.testing.assert_allclose(ref, hyp_jit)
    return mod


class PositionalConvEmbedding(Module):
    def __init__(
            self,
            in_channels: int,
            num_embeddings: int,
            num_embedding_groups: int,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=num_embeddings,
            padding=num_embeddings // 2,
            groups=num_embedding_groups,
        )
        # self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        self.num_remove: int = 1 if num_embeddings % 2 == 0 else 1

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., :-self.num_remove]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(1, 2)
        return x


def _import_pos_conv(module: Module, debug: bool):
    assert module.activation is torch.nn.functional.gelu
    mod = PositionalConvEmbedding(
        in_channels=module.conv.in_channels,
        num_embeddings=module.conv.kernel_size[0],
        num_embedding_groups=module.conv.groups
    )
    torch.nn.utils.remove_weight_norm(module.conv)
    mod.load_state_dict(module.state_dict())
    if debug:
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)

        input = torch.randn(3, 10, module.conv.in_channels)
        ref = original(input)
        hyp = imported(input)
        hyp_jit = scripted(input)
        torch.testing.assert_allclose(ref, hyp)
        torch.testing.assert_allclose(ref, hyp_jit)
    return mod


# TODO: Replace this with torch.nn.MultiheadAttention
class Attention(Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
    ):
        super().__init__() 
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,
    ):
        # assumption output_attentions is always True
        batch_size, tgt_len, embed_dim = x.size()

        query_states = self.q_proj(x) * self.scaling
        key_states = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = query_states.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, tgt_len, src_len)

        attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (batch_size * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


def _import_attention(module: Module, debug: bool):
    assert not module.is_decoder
    assert module.k_proj.bias is not None
    assert module.v_proj.bias is not None
    assert module.q_proj.bias is not None
    mod = Attention(
        embed_dim=module.embed_dim,
        num_heads=module.num_heads,
        dropout=module.dropout,
    )
    mod.load_state_dict(module.state_dict())
    if debug:
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)

        s, l, n, e = 16, 16, 3, module.embed_dim
        query = torch.randn(l, n, e)

        ref_out, ref_weight, _ = original(query, attention_mask=None, output_attentions=True)
        hyp_out, hyp_weight = imported(query)
        hyp_out_jit, hyp_weight_jit = scripted(query)
        torch.testing.assert_allclose(ref_out, hyp_out)
        torch.testing.assert_allclose(ref_out, hyp_out_jit)
        torch.testing.assert_allclose(ref_weight, hyp_weight)
        torch.testing.assert_allclose(ref_weight, hyp_weight_jit)
    return mod


class FeedForward(Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            intermediate_dropout: float,
            output_dropout: float,
    ):
        super().__init__()
        self.intermediate_dense = nn.Linear(in_features, out_features)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = nn.Linear(out_features, in_features)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x):
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x


def _import_feed_forward(module, debug):
    assert module.intermediate_act_fn is torch.nn.functional.gelu
    mod = FeedForward(
        in_features=module.intermediate_dense.in_features,
        out_features=module.intermediate_dense.out_features,
        intermediate_dropout=module.intermediate_dropout.p,
        output_dropout=module.output_dropout.p,
    )
    mod.load_state_dict(module.state_dict())
    if debug:
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)

        input = torch.randn(3, 10, mod.intermediate_dense.in_features)
        ref = original(input)
        hyp = imported(input)
        hyp_jit = scripted(input)
        torch.testing.assert_allclose(ref, hyp)
        torch.testing.assert_allclose(ref, hyp_jit)
    return mod


class EncoderLayer(Module):
    def __init__(
            self,
            attention: Module,
            dropout: float,
            normalized_shape: int,
            layer_norm_eps: float,
            layer_norm_first: bool,
            feed_forward: Module,
    ):
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=layer_norm_eps)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(normalized_shape, eps=layer_norm_eps)

    def forward(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,
    ):
        attn_residual = x
        if self.layer_norm_first:
            x = self.layer_norm(x)

        x, attn_weights = self.attention(x, attention_mask)
        x = self.dropout(x)
        x = attn_residual + x

        if self.layer_norm_first:
            x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x)
            x = self.final_layer_norm(x + self.feed_forward(x))
        return x, attn_weights


def _import_encoder_layer(module: Module, debug: bool, indent: int):
    mod = EncoderLayer(
        attention=_import(module.attention, debug, indent),
        dropout=module.dropout.p,
        normalized_shape=module.layer_norm.normalized_shape,
        layer_norm_eps=module.layer_norm.eps,
        layer_norm_first=False,
        feed_forward=_import(module.feed_forward, debug, indent),
    )
    mod.load_state_dict(module.state_dict())
    if debug:
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)

        s, l, n, e = 16, 16, 3, module.attention.embed_dim
        query = torch.randn(l, n, e)

        ref_out, ref_weight = original(query, attention_mask=None, output_attentions=True)
        hyp_out, hyp_weight = imported(query)
        hyp_out_jit, hyp_weight_jit = scripted(query)
        torch.testing.assert_allclose(ref_out, hyp_out)
        torch.testing.assert_allclose(ref_out, hyp_out_jit)
        torch.testing.assert_allclose(ref_weight, hyp_weight)
        torch.testing.assert_allclose(ref_weight, hyp_weight_jit)
    return mod


def _import_encoder_layer_stable_layer_norm(module, debug, indent):
    mod = EncoderLayer(
        attention=_import(module.attention, debug, indent),
        dropout=module.dropout.p,
        normalized_shape=module.layer_norm.normalized_shape,
        layer_norm_eps=module.layer_norm.eps,
        layer_norm_first=True,
        feed_forward=_import(module.feed_forward, debug, indent),
    )
    mod.load_state_dict(module.state_dict())
    if debug:
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)

        s, l, n, e = 16, 16, 3, module.attention.embed_dim
        query = torch.randn(l, n, e)

        ref_out, ref_weight = original(query, attention_mask=None, output_attentions=True)
        hyp_out, hyp_weight = imported(query)
        hyp_out_jit, hyp_weight_jit = scripted(query)
        torch.testing.assert_allclose(ref_out, hyp_out)
        torch.testing.assert_allclose(ref_out, hyp_out_jit)
        torch.testing.assert_allclose(ref_weight, hyp_weight)
        torch.testing.assert_allclose(ref_weight, hyp_weight_jit)
    return mod


class Encoder(Module):
    def __init__(
            self,
            pos_conv_embed: Module,
            normalized_shape,
            layer_norm_eps,
            layer_norm_first: bool,
            dropout: float,
            layers: Module,
    ):
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=layer_norm_eps)
        self.layer_norm_first = layer_norm_first
        self.dropout = nn.Dropout(dropout)
        self.layers = layers

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None):
        if attention_mask is not None:
            # make sure padded tokens output 0
            x[~attention_mask] = 0.0

            # extend attention_mask
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=x.dtype)) * -10000.0
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        x = x + self.pos_conv_embed(x)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.dropout(x)
        for layer in self.layers:
            x, _ = layer(x, attention_mask)

        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x


def _import_encoder(module: Module, layer_norm_first: bool, debug: bool, indent: int):
    mod = Encoder(
        pos_conv_embed=_import(module.pos_conv_embed, debug, indent),
        normalized_shape=module.layer_norm.normalized_shape,
        layer_norm_eps=module.layer_norm.eps,
        layer_norm_first=layer_norm_first,
        dropout=module.dropout.p,
        layers=_import(module.layers, debug, indent),
    )
    mod.load_state_dict(module.state_dict())
    if debug:
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)

        s, l, n, e = 16, 16, 3, mod.pos_conv_embed.conv.in_channels
        query = torch.randn(l, n, e)

        ref_out = original(query).last_hidden_state
        hyp_out = imported(query)
        hyp_out_jit = scripted(query)
        torch.testing.assert_allclose(ref_out, hyp_out)
        torch.testing.assert_allclose(ref_out, hyp_out_jit)
    return mod


class Wav2Vec2Model(Module):
    def __init__(
            self,
            feature_extractor: Module,
            feature_projection: Module,
            encoder: Module,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_projection = feature_projection
        self.encoder = encoder

    def forward(
            self,
            waveforms: Tensor,
            attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.feature_extractor(waveforms)
        x = x.transpose(1, 2)

        if attention_mask is not None:
            output_lengths = self.feature_extractor.get_output_length(attention_mask.sum(-1))
            attention_mask = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)
            index = torch.arange(attention_mask.shape[0], device=x.device)
            attention_mask[index, output_lengths - 1] = 1
            attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).to(torch.bool)
        x = self.feature_projection(x)

        x = self.encoder(x, attention_mask=attention_mask)
        return x


def _import_w2v2_model(module, debug, indent):
    mod = Wav2Vec2Model(
        feature_extractor=_import(module.feature_extractor, debug, indent),
        feature_projection=_import(module.feature_projection, debug, indent),
        encoder=_import(module.encoder, debug, indent),
    )
    if debug:
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)

        input = torch.randn(3, 1024)

        ref_out = original(input).last_hidden_state
        hyp_out = imported(input)
        hyp_out_jit = scripted(input)
        torch.testing.assert_allclose(ref_out, hyp_out)
        torch.testing.assert_allclose(ref_out, hyp_out_jit)
    return mod


class Wav2Vec2ForCTC(Module):
    def __init__(
            self,
            wav2vec2_model: Module,
            dropout: float,
            in_features: int,
            out_features: int,
    ):
        super().__init__()
        self.wav2vec2 = wav2vec2_model
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(in_features, out_features)

    def forward(
            self,
            waveform: Tensor,
            attention_mask: Optional[Tensor] = None,
    ):
        x = self.wav2vec2(waveform, attention_mask)
        x = self.dropout(x)
        logits = self.lm_head(x)
        return logits


def _import_w2v2_ctc(module, debug, indent):
    mod = Wav2Vec2ForCTC(
        wav2vec2_model=_import(module.wav2vec2, debug, indent),
        dropout=module.dropout.p,
        in_features=module.lm_head.in_features,
        out_features=module.lm_head.out_features,
    )
    blacklist = ['wav2vec2.masked_spec_embed']
    state_dict = {k: v for k, v in module.state_dict().items() if k not in blacklist}
    mod.load_state_dict(state_dict)
    if debug:
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)

        input = torch.randn(3, 1024)

        ref_out = original(input).logits
        hyp_out = imported(input)
        hyp_out_jit = scripted(input)
        torch.testing.assert_allclose(ref_out, hyp_out)
        torch.testing.assert_allclose(ref_out, hyp_out_jit)
    return mod


def _import(module: Module, debug, indent):
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Attention,
        Wav2Vec2Encoder,
        Wav2Vec2EncoderLayer,
        Wav2Vec2EncoderLayerStableLayerNorm,
        Wav2Vec2EncoderStableLayerNorm,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2FeatureProjection,
        Wav2Vec2FeedForward,
        Wav2Vec2ForCTC,
        Wav2Vec2GroupNormConvLayer,
        Wav2Vec2LayerNormConvLayer,
        Wav2Vec2Model as Wav2Vec2Model_,
        Wav2Vec2NoLayerNormConvLayer,
        Wav2Vec2PositionalConvEmbedding,
    )

    spaces = '  ' * indent
    indent += 1

    # Handle torch's module containers
    if isinstance(module, torch.nn.Sequential):
        _LG.debug('%s- Importing torch.nn.Sequential', spaces)
        return torch.nn.Sequential(*[_import(m, debug, indent) for m in module])
    if isinstance(module, torch.nn.ModuleList):
        _LG.debug('%s- Importing torch.nn.ModuleList', spaces)
        return torch.nn.ModuleList([_import(m, debug, indent) for m in module])

    # Handle huggingface's custom modules
    if isinstance(module, Wav2Vec2ForCTC):
        _LG.info('%s* Importing Wav2Vec2ForCTC', spaces)
        return _import_w2v2_ctc(module, debug, indent)
    if isinstance(module, Wav2Vec2Model_):
        _LG.info('%s* Importing Wav2Vec2Model', spaces)
        return _import_w2v2_model(module, debug, indent)
    if isinstance(module, Wav2Vec2FeatureExtractor):
        _LG.info('%s* Importing Wav2Vec2FeatureExtractor', spaces)
        return _import_feature_extractor(module, debug, indent)
    if isinstance(module, Wav2Vec2GroupNormConvLayer):
        _LG.info('%s* Importing Wav2Vec2GroupNormConvLayer', spaces)
        return _import_conv_layer(module, "group", debug)
    if isinstance(module, Wav2Vec2LayerNormConvLayer):
        _LG.info('%s* Importing Wav2Vec2LayerNormConvLayer', spaces)
        return _import_conv_layer(module, "layer", debug)
    if isinstance(module, Wav2Vec2NoLayerNormConvLayer):
        _LG.info('%s* Importing Wav2Vec2NoLayerNormConvLayer', spaces)
        return _import_conv_layer(module, None, debug)
    if isinstance(module, Wav2Vec2FeatureProjection):
        _LG.info('%s* Importing Wav2Vec2FeatureProjection', spaces)
        return _import_feature_projection(module, debug)
    if isinstance(module, Wav2Vec2Encoder):
        _LG.info('%s* Importing Wav2Vec2Encoder', spaces)
        return _import_encoder(module, True, debug, indent)
    if isinstance(module, Wav2Vec2EncoderStableLayerNorm):
        _LG.info('%s* Importing Wav2Vec2EncoderStableLayerNorm', spaces)
        return _import_encoder(module, False, debug, indent)
    if isinstance(module, Wav2Vec2PositionalConvEmbedding):
        _LG.info('%s* Importing Wav2Vec2PositionalConvEmbedding', spaces)
        return _import_pos_conv(module, debug)
    if isinstance(module, Wav2Vec2EncoderLayer):
        _LG.info('%s* Importing Wav2Vec2EncoderLayer', spaces)
        return _import_encoder_layer(module, debug, indent)
    if isinstance(module, Wav2Vec2EncoderLayerStableLayerNorm):
        _LG.info('%s* Importing Wav2Vec2EncoderLayerStableLayerNorm', spaces)
        return _import_encoder_layer_stable_layer_norm(module, debug, indent)
    if isinstance(module, Wav2Vec2Attention):
        _LG.info('%s* Importing Wav2Vec2Attention', spaces)
        return _import_attention(module, debug)
    if isinstance(module, Wav2Vec2FeedForward):
        _LG.info('%s* Importing Wav2Vec2FeedForward', spaces)
        return _import_feed_forward(module, debug)

    raise NotImplementedError(f'Conversion not supported: {module.__class__}')


@_mod_utils.requires_module('transformers')
def import_huggingface_model(model: Module, debug: bool = False) -> Module:
    """

    Note:
        This function modifies the original model.
    """
    return _import(model, debug, indent=0)
