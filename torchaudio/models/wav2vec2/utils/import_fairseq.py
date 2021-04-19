"""Import fariseq's wav2vec2.0 pretrained weights to torchaudios's format.

For this module to work, you need `fairseq`.
"""

import logging

import torch
from torch.nn import Module

from torchaudio._internal import module_utils as _mod_utils
from ..components import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    SamePad,
    TransposeLast,
    ConvFeatureExtractionModel,
    TransformerSentenceEncoderLayer,
    TransformerEncoder,
    Wav2Vec2Model,
    Wav2VecEncoder,
    Wav2VecCtc,
)


_LG = logging.getLogger(__name__)


def _test_eval_compatibility(original, imported, shape):
    original = original.eval()
    imported = imported.eval()
    scripted = torch.jit.script(imported)

    input = torch.randn(shape)
    reference = original(input.clone())
    result_eager = imported(input.clone())
    result_jit = scripted(input.clone())
    torch.testing.assert_allclose(reference, result_eager)
    torch.testing.assert_allclose(reference, result_jit)


def _import_mha(module: Module, debug: bool):
    mod = torch.nn.MultiheadAttention(
        embed_dim=module.embed_dim,
        num_heads=module.num_heads,
        dropout=module.dropout_module.p,
        bias=module.k_proj.bias is not None,
        add_bias_kv=module.bias_k is not None,
        add_zero_attn=module.add_zero_attn,
        kdim=module.kdim,
        vdim=module.vdim,
    )
    # fairseq uses separate weights/biases for query/key/value
    # PyTorch uses fused weight/bias when embed_dim == kdim == vdim
    state_dict = module.state_dict()
    if module.embed_dim == module.kdim == module.vdim:
        blacklist = [f'{k}_proj.bias' for k in 'qkv'] + [f'{k}_proj.weight' for k in 'qkv']
        state_dict_ = {k: v for k, v in state_dict.items() if k not in blacklist}
        state_dict_['in_proj_weight'] = torch.cat([state_dict[f'{k}_proj.weight'] for k in 'qkv'])
        state_dict_['in_proj_bias'] = torch.cat([state_dict[f'{k}_proj.bias'] for k in 'qkv'])
        state_dict = state_dict_
    mod.load_state_dict(state_dict)

    if debug:
        original = module.eval()
        imported = mod.eval()
        s, l, n, e = 250, 250, 32,  module.embed_dim
        query, key, value = torch.randn(l, n, e), torch.randn(s, n, e), torch.randn(s, n, e)
        # Test numerical compatibility
        ref_attn_output, ref_attn_output_weights = original(query, key, value)
        res_attn_output, res_attn_output_weights = imported(query, key, value)
        torch.testing.assert_allclose(ref_attn_output, res_attn_output)
        torch.testing.assert_allclose(ref_attn_output_weights, res_attn_output_weights)
        # Test TorchScript
        scripted = torch.jit.script(imported)
        res_attn_output, res_attn_output_weights = scripted(query, key, value)
        torch.testing.assert_allclose(ref_attn_output, res_attn_output)
        torch.testing.assert_allclose(ref_attn_output_weights, res_attn_output_weights)
    return mod


def _import_tse(module, indent, debug):
    assert module.activation_fn.__name__ == 'gelu'
    mod = TransformerSentenceEncoderLayer(
        self_attn=_import(module.self_attn, indent, debug),
        dropout1=_import(module.dropout1, indent, debug),
        dropout2=_import(module.dropout2, indent, debug),
        dropout3=_import(module.dropout3, indent, debug),
        self_attn_layer_norm=_import(module.self_attn_layer_norm, indent, debug),
        fc1=_import(module.fc1, indent, debug),
        fc2=_import(module.fc2, indent, debug),
        final_layer_norm=_import(module.final_layer_norm, indent, debug),
        layer_norm_first=module.layer_norm_first,
    )
    if debug:
        original = module.eval()
        imported = mod.eval()
        input = torch.randn(250, 32, module.self_attn.embed_dim)
        # Test numerical compatibility
        ref_x, ref_attn = original(input)
        res_x, res_attn = imported(input)
        torch.testing.assert_allclose(ref_x, res_x)
        assert ref_attn is None and res_attn is None
        # Test TorchScript
        scripted = torch.jit.script(imported)
        res_x, res_attn = scripted(input)
        torch.testing.assert_allclose(ref_x, res_x)
        assert ref_attn is None and res_attn is None
    return mod


def _import_w2v2model(module, indent, debug):
    mod = Wav2Vec2Model(
        feature_extractor=_import(module.feature_extractor, indent, debug),
        post_extract_proj=_import(module.post_extract_proj, indent, debug),
        dropout_input=_import(module.dropout_input, indent, debug),
        dropout_features=_import(module.dropout_features, indent, debug),
        encoder=_import(module.encoder, indent, debug),
        layer_norm=_import(module.layer_norm, indent, debug),
    )
    if debug:
        # Test extract_features method
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)
        sample = torch.randn(2, 1024)
        padding_mask = torch.randn_like(sample) > 0
        # with mask=None
        ref_x, ref_mask = original.extract_features(sample, padding_mask=None, mask=False)
        res1_x, res1_mask = imported.extract_features(sample, padding_mask=None)
        res2_x, res2_mask = scripted.extract_features(sample, padding_mask=None)
        torch.testing.assert_allclose(ref_x, res1_x)
        torch.testing.assert_allclose(ref_x, res2_x)
        assert ref_mask is None and res1_mask is None and res2_mask is None
        # with mask
        ref_x, ref_mask = original.extract_features(sample, padding_mask=padding_mask, mask=False)
        res1_x, res1_mask = imported.extract_features(sample, padding_mask=padding_mask)
        res2_x, res2_mask = scripted.extract_features(sample, padding_mask=padding_mask)
        torch.testing.assert_allclose(ref_x, res1_x)
        torch.testing.assert_allclose(ref_x, res2_x)
        torch.testing.assert_allclose(ref_mask, res1_mask)
        torch.testing.assert_allclose(ref_mask, res2_mask)
    return mod


def _import_w2v2encoder(module, indent, debug):
    mod = Wav2VecEncoder(
        w2v_model=_import(module.w2v_model, indent, debug),
        final_dropout=_import(module.final_dropout, indent, debug),
        proj=_import(module.proj, indent, debug),
    )
    if debug:
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)
        sample = torch.randn(2, 1024)
        padding_mask = torch.randn_like(sample) > 0
        ref = original(sample, padding_mask)
        res0 = imported(sample, padding_mask)
        res1 = scripted(sample, padding_mask)
        for key in ['encoder_out', 'encoder_padding_mask', 'padding_mask']:
            torch.testing.assert_allclose(ref[key], res0[key])
            torch.testing.assert_allclose(ref[key], res1[key])
    return mod


def _import_w2v2ctc(module, indent, debug):
    mod = Wav2VecCtc(
        w2v_encoder=_import(module.w2v_encoder, indent, debug),
    )
    if debug:
        original = module.eval()
        imported = mod.eval()
        scripted = torch.jit.script(imported)
        sample = torch.randn(2, 1024)
        padding_mask = torch.randn_like(sample) > 0
        ref = original(source=sample, padding_mask=padding_mask)
        res0 = imported(source=sample, padding_mask=padding_mask)
        res1 = scripted(source=sample, padding_mask=padding_mask)
        for key in ['encoder_out', 'encoder_padding_mask', 'padding_mask']:
            torch.testing.assert_allclose(ref[key], res0[key])
            torch.testing.assert_allclose(ref[key], res1[key])
    return mod


def _import(module: Module, indent, debug):
    from fairseq.modules.fp32_group_norm import Fp32GroupNorm as Fp32GroupNorm_
    from fairseq.modules.layer_norm import Fp32LayerNorm as Fp32LayerNorm_
    from fairseq.modules.same_pad import SamePad as SamePad_
    from fairseq.modules.transpose_last import TransposeLast as TransposeLast_
    from fairseq.modules.multihead_attention import MultiheadAttention as MultiheadAttention_
    from fairseq.models.wav2vec.wav2vec2 import (
        ConvFeatureExtractionModel as ConvFeatureExtractionModel_,
        TransformerEncoder as TransformerEncoder_,
        TransformerSentenceEncoderLayer as TransformerSentenceEncoderLayer_,
        Wav2Vec2Model as Wav2Vec2Model_,
    )
    from fairseq.models.wav2vec.wav2vec2_asr import (
        Wav2VecCtc as Wav2VecCtc_,
        Wav2VecEncoder as Wav2VecEncoder_,
    )

    spaces = '  ' * indent
    indent += 1

    # Handle fairseq's custom modules
    if isinstance(module, Fp32GroupNorm_):
        _LG.info('%s* Importing Fp32GroupNorm', spaces)
        mod = Fp32GroupNorm(
            num_groups=module.num_groups,
            num_channels=module.num_channels,
            eps=module.eps,
            affine=module.affine,
        )
        mod.load_state_dict(module.state_dict())
        if debug:
            _test_eval_compatibility(module, mod, (20, module.num_channels, 10, 10))
        return mod
    if isinstance(module, Fp32LayerNorm_):
        _LG.info('%s* Importing Fp32LayerNorm', spaces)
        mod = Fp32LayerNorm(
            normalized_shape=module.normalized_shape,
            eps=module.eps,
            elementwise_affine=module.elementwise_affine,
        )
        mod.load_state_dict(module.state_dict())
        if debug:
            _test_eval_compatibility(module, mod, (20, module.bias.numel()))
        return mod
    if isinstance(module, SamePad_):
        _LG.info('%s* Importing SamePad', spaces)
        mod = SamePad(0)
        mod.remove = module.remove
        if debug:
            _test_eval_compatibility(module, mod, (2, 3, module.remove * 2))
        return mod
    if isinstance(module, TransposeLast_):
        _LG.info('%s* Importing TransposeLast', spaces)
        mod = TransposeLast()
        return mod
    if isinstance(module, MultiheadAttention_):
        _LG.info('%s* Importing MultiheadAttention', spaces)
        return _import_mha(module, debug)
    if isinstance(module, ConvFeatureExtractionModel_):
        _LG.info('%s* Importing ConvFeatureExtractionModel', spaces)
        mod = ConvFeatureExtractionModel(
            conv_layers=_import(module.conv_layers, indent, debug)
        )
        if debug:
            _test_eval_compatibility(module, mod, (2, 1024))
        return mod
    if isinstance(module, TransformerSentenceEncoderLayer_):
        _LG.info('%s* Importing TransformerSentenceEncoderLayer', spaces)
        return _import_tse(module, indent, debug)
    if isinstance(module, TransformerEncoder_):
        _LG.info('%s* Importing TransformerEncoder', spaces)
        mod = TransformerEncoder(
            pos_conv=_import(module.pos_conv, indent, debug),
            layers=_import(module.layers, indent, debug),
            layer_norm=_import(module.layer_norm, indent, debug),
            dropout=module.dropout,
            layer_norm_first=module.layer_norm_first,
        )
        if debug:
            _LG.debug('%s+ Testing TransformerEncoder', spaces)
            _test_eval_compatibility(module, mod, (1, 2, module.embedding_dim))
        return mod
    if isinstance(module, Wav2Vec2Model_):
        _LG.info('%s* Importing Wav2Vec2Model', spaces)
        return _import_w2v2model(module, indent, debug)
    if isinstance(module, Wav2VecEncoder_):
        _LG.info('%s* Importing Wav2VecEncoder', spaces)
        return _import_w2v2encoder(module, indent, debug)
    if isinstance(module, Wav2VecCtc_):
        _LG.info('%s* Importing Wav2VecCtc', spaces)
        return _import_w2v2ctc(module, indent, debug)

    # Handle torch's native modules
    if isinstance(module, torch.nn.Sequential):
        _LG.debug('%s- Importing torch.nn.Sequential', spaces)
        return torch.nn.Sequential(*[_import(m, indent, debug) for m in module])
    if isinstance(module, torch.nn.ModuleList):
        _LG.debug('%s- Importing torch.nn.ModuleList', spaces)
        return torch.nn.ModuleList([_import(m, indent, debug) for m in module])
    # TODO: clone module by properly constructing the Module object.
    if module.__class__.__module__.startswith('torch.nn'):
        _LG.debug('%s- %s.%s', spaces, module.__class__.__module__, module.__class__.__name__)
        # Modules with weight_norm hook are not script-able. So we remove the hook.
        # https://github.com/pytorch/pytorch/issues/57289
        # https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/models/wav2vec/wav2vec2.py#L765
        for hook in module._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if (
                    hook.__module__ == 'torch.nn.utils.weight_norm' and
                    hook.__class__.__name__ == 'WeightNorm'
            ):
                _LG.warning('%s! Removing weight_norm from %s', spaces, module)
                torch.nn.utils.remove_weight_norm(module)
        # Check TS support
        torch.jit.script(module)
        return module

    raise NotImplementedError(f'Conversion not supported: {module.__class__}')


@_mod_utils.requires_module('fairseq')
def import_fairseq_model(model: Module, debug: bool = False) -> Module:
    """Import fairseq's wav2vec2.0 model into torchaudio's format

    Args:
        model (torch.nn.Module):
            Model such as Wav2Vec2Model, Wav2Vec2Ctc and Wav2Vec2Encoder
        debug (bool, optional):
            When enabled, each individual modules are tested for numerical compatibility
            against the original module in eager mode and TorchScript mode.

    Returns:
        torch.nn.Module:
            An instance of the corresponding model class.

    Example:
        >>> model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        ...     [checkpoint_path], arg_overrides={'data': data_dir})
        >>> imported = import_fairseq_model(model[0])

    Note:
        At the moment, the internal structure of the model is different from the original,
        thus torchaudio's model exhibits different behavior than `fairseq` model.
        - `fairseq`'s `TransformerEncoder` applies random layer drop at training,
           but `torchaudio`'s `TransformerEncoder` does not.
        - A convolution layer in `fairseq`'s `TransformerEncoder` has weight normalization hook,
           but `torchaudio`'s `TransformerEncoder` does not.
    """
    model_ = _import(model, indent=0, debug=debug)
    model_.load_state_dict(model.state_dict(), strict=False)
    return model_
