import torch.nn as nn
import einops
import torch
from transformers.models.t5.modeling_t5 import (
    T5LayerFF, T5Block,
    T5LayerSelfAttention, T5Attention,T5DenseActDense
)
from transformers import T5Config
from typing import Optional
from transformers.activations import ACT2FN
import math



class TimeConvAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        kernel_size = (self.config.am_adapter_kernel_size_t, self.config.am_adapter_kernel_size_l)
        in_channels, adapter_channels = self.config.d_model, self.config.am_adapter_channels

        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv2d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)

        self.layer_norm = nn.LayerNorm(in_channels)


    def forward(self, x, attention_mask):
        residual = x
        x = einops.rearrange(x,'(b n) l d -> b n l d', n=self.config.num_frames)
        x = self.fc1(x)

        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = einops.rearrange(x, 'b d n l  -> b n l d', n=self.config.num_frames)

        x = self.fc2(x)
        x = self.layer_norm(x)
        x = einops.rearrange(x, 'b n l d -> (b n) l d')

        return x + residual


class TimeGatingUnit(nn.Module):
    """
    Time Gate
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        d_z, seq_len = config.d_ff, config.num_frames

        self.norm = nn.LayerNorm([d_z // 2])
        self.weight = nn.Parameter(torch.zeros(seq_len, seq_len).uniform_(-0.01, 0.01), requires_grad=True)
        self.bias = nn.Parameter(torch.ones(seq_len), requires_grad=True)


    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        note that seq_len is num_frames
        :param z: (batch_size, num frames) * len * d_ff
        :param mask: None
        :return: (batch_size, num frames) * len * d_ff//2
        """

        z = einops.rearrange(z,'(b n) l d_ff -> n (b l) d_ff', n=self.config.num_frames)

        seq_len = z.shape[0]

        z1, z2 = torch.chunk(z, 2, dim=-1)

        # Check mask
        if mask is not None:
            assert mask.shape[0] == 1 or mask.shape[0] == seq_len
            assert mask.shape[1] == seq_len
            assert mask.shape[2] == 1
            mask = mask[:, :, 0]

        z2 = self.norm(z2)
        weight = self.weight[:seq_len, :seq_len]
        if mask is not None:
            weight = weight * mask

        z2 = torch.einsum('ij,jbd->ibd', weight, z2) + self.bias[:seq_len, None, None]

        output = z1 * z2
        output = einops.rearrange(output,'n (b l) d -> (b n) l d', n=self.config.num_frames, l=self.config.ocr_len + self.config.obj_len)
        return output


##################魔改ATTN 考虑 spatial begin #############################

# 魔改Attn，考虑spatial
class SpatialAttention(T5Attention):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__(config=config, has_relative_attention_bias=has_relative_attention_bias)
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)



    def forward(
            self,
            hidden_states,
            mask=None,
            key_value_states=None,
            position_bias=None,
            past_key_value=None,
            layer_head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False,
            bbox_embeds=None,

    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias
        scores += position_bias_masked


        ######### modify spatial attention #########
        # bbox positional encoding
        batch_size, n_head, seq_length, d_head = query_states.shape
        bbox_pos_emb = bbox_embeds.view(seq_length, seq_length, batch_size, d_head)
        bbox_pos_emb = bbox_pos_emb.permute([2, 0, 1, 3])
        bbox_pos_scores = torch.einsum("bnid,bijd->bnij", (query_states, bbox_pos_emb))


        scores += bbox_pos_scores
        scores = scores / math.sqrt(self.attention_head_size)
        ###########################################

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs



class BrosLayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config=config, has_relative_attention_bias=has_relative_attention_bias)
        # modify
        self.SelfAttention = SpatialAttention(config, has_relative_attention_bias=has_relative_attention_bias)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        bbox_embeds=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            bbox_embeds=bbox_embeds, # add
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


##################魔改ATTN 考虑 spatial  end#############################
class VideoT5EncoderLayer(nn.Module):
    def __init__(self, config, num_layer,has_relative_attention_bias=False, ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()

        self.config = config

        # temporal
        self.time_conv_adapter = TimeConvAdapter(config)
        # spatial
        self.layer.append(BrosLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        # FFN
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        bbox_embeds=None,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (key / value) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        # modify
        # time conv adapter
        hidden_states = self.time_conv_adapter(hidden_states, attention_mask)

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            bbox_embeds=bbox_embeds
        )

        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)

