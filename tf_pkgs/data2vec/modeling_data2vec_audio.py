# coding=utf-8
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Data2VecAudio model."""

import math
import warnings
from typing import Optional, Tuple, Union, Dict, List, Any

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_peft_available,
    logging,
)
from transformers.models.data2vec.configuration_data2vec_audio import Data2VecAudioConfig


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 2

# General docstring
_CONFIG_FOR_DOC = "Data2VecAudioConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/data2vec-audio-base-960h"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

# CTC docstring
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 66.95


# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


class Data2VecAudioConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PadLayer
class Data2VecAudioPadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class Data2VecAudioPositionalConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_pos_kernel_size,
            padding=config.conv_pos_kernel_size // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        self.padding = Data2VecAudioPadLayer(config.conv_pos_kernel_size)
        self.activation = ACT2FN[config.feat_extract_activation]
        # no learnable parameters
        self.layer_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding
class Data2VecAudioPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        self.padding = Data2VecAudioPadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Custom implementation without requires_grad setting
class Data2VecAudioFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Handle missing config attributes with fallbacks
        # These should exist in the original model but might be missing in the loaded config
        feat_extract_norm = getattr(config, "feat_extract_norm", "group")
        conv_groups = getattr(config, "conv_groups", [1] * (len(config.conv_dim) - 1))

        self.conv_layers = nn.ModuleList()
        if feat_extract_norm == "group":
            # First conv layer handles the raw waveform (1 channel) to config.conv_dim[0] (512)
            # The rest follow the config pattern
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=1,  # Raw audio has 1 channel
                    out_channels=config.conv_dim[0],
                    kernel_size=config.conv_kernel[0],
                    stride=config.conv_stride[0],
                )
            )
            self.conv_layers.append(nn.GroupNorm(num_groups=1, num_channels=config.conv_dim[0]))
            self.conv_layers.append(ACT2FN[config.feat_extract_activation])

            # Add the rest of the layers as in the original implementation
            conv_layers = [
                (config.conv_dim[i], config.conv_dim[i + 1], config.conv_kernel[i+1], config.conv_stride[i+1])
                for i in range(len(config.conv_dim) - 1)
            ]
            for i, (in_channels, out_channels, kernel, stride) in enumerate(conv_layers):
                group_idx = min(i, len(conv_groups) - 1)
                group_size = conv_groups[group_idx] if conv_groups is not None else 1
                
                conv = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    groups=group_size,
                )
                self.conv_layers.append(conv)
                self.conv_layers.append(nn.GroupNorm(num_groups=group_size, num_channels=out_channels))
                self.conv_layers.append(ACT2FN[config.feat_extract_activation])
        else:
            raise ValueError(
                f"`feat_extract_norm` is {feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self._requires_grad = False  # Always set to False for inference

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        # Ensure the input has the right shape [batch_size, sequence_length]
        if input_values.dim() == 1:
            # Add batch dimension if input is just a single sequence
            input_values = input_values.unsqueeze(0)
            
        # Add channel dimension for conv1d [batch_size, channels, sequence_length]
        hidden_states = input_values.unsqueeze(1) 
        
        # Forward through all feature extraction layers
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection
class Data2VecAudioFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


# Simpler attention implementation that is compatible with TorchScript
class Data2VecAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        """Reshape for multi-head attention computation with explicit type annotations for TorchScript"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # We never need attention outputs for inference
        output_attentions = False

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        
        # For TorchScript compatibility, handle possible null states
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            # Fix for TorchScript: Ensure key_value_states is not None before passing to k_proj and v_proj
            if key_value_states is None:
                # This shouldn't happen in normal use, but TorchScript needs explicit handling
                # We'll fall back to using hidden_states as both query and key/value
                key_value_states = hidden_states
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # For inference, we don't need to keep track of past key values
        past_key_value_out = None
        
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # For inference, we don't need to handle layer_head_mask
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # For inference, we never need attention weights
        return attn_output, None, past_key_value_out


# TorchScript-compatible SDPA attention without logger.warning
class Data2VecAudioSdpaAttention(Data2VecAudioAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # Always set output_attentions to False for inference
        output_attentions = False
        
        # No additional handling needed here since the parent class already has the fix
        # and we call super().forward() with the same parameters
        
        # Always use the parent's implementation to avoid scaled_dot_product_attention issues
        return super().forward(
            hidden_states,
            key_value_states=key_value_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )


# Dictionary for attention implementations
DATA2VEC2AUDIO_ATTENTION_CLASSES = {
    "eager": Data2VecAudioAttention,
    "sdpa": Data2VecAudioSdpaAttention,
}


# Feed-forward implementation from the model
class Data2VecAudioFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# Encoder layer implementation
class Data2VecAudioEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DATA2VEC2AUDIO_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Data2VecAudioFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # We never need attention outputs for inference
        output_attentions = False
        
        attn_residual = hidden_states
        hidden_states, _, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=False
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        # Always return a tuple with hidden states and None for attention weights
        return (hidden_states, None)


# Encoder implementation with explicit loops instead of list comprehensions
class Data2VecAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Data2VecAudioPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([Data2VecAudioEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # For inference-only model, we never need these outputs
        output_attentions = False
        output_hidden_states = False
        
        if attention_mask is not None:
            # make sure padded tokens are not attended to
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            
            # extend attention_mask
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Forward through all encoder layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                # Always set this to False for inference
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]

        # For inference, always return dictionary
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": None,
            "attentions": None
        }


# Base model class with minimal dependencies
class Data2VecAudioPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Data2VecAudioConfig
    base_model_prefix = "data2vec_audio"
    main_input_name = "input_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)


# Main model implementation
class Data2VecAudioModel(Data2VecAudioPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Data2VecAudioFeatureEncoder(config)
        self.feature_projection = Data2VecAudioFeatureProjection(config)
        self.encoder = Data2VecAudioEncoder(config)
        
        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # For inference, we never need these options
        output_attentions = False
        output_hidden_states = False
        return_dict = True  # Always return dict for TorchScript compatibility

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask
            )

        hidden_states, _ = self.feature_projection(extract_features)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # Return a dictionary structure for TorchScript compatibility
        return {
            "last_hidden_state": encoder_outputs["last_hidden_state"],
            "extract_features": extract_features,
            "hidden_states": None,
            "attentions": None,
        }
        
    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_zero_indexes = attention_mask.sum(dim=-1, keepdim=True).long()
        output_lengths = torch.clamp(non_zero_indexes, min=0, max=feature_vector_length)
        output_lengths = output_lengths.squeeze(-1)

        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths indices are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask
