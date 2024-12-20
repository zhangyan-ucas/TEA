import einops
from transformers import T5Config, T5ForConditionalGeneration, CLIPModel, AutoConfig
import torch.nn as nn
import numpy as np
from transformers.models.t5.modeling_t5 import (
    T5LayerNorm, T5Stack,
    T5PreTrainedModel, T5Model,
    T5EncoderModel,  T5DenseActDense,
    T5Attention, T5DenseGatedActDense,
    T5LayerSelfAttention, T5LayerCrossAttention,
    T5LayerFF, T5Block
)
import matplotlib.pyplot as plt
import copy
from typing import Optional, Tuple, Union
import torch
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
)
from transformers.file_utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.activations import ACT2FN

from torch.nn import CrossEntropyLoss
import warnings
from transformers.utils import logging
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from transformers import AutoConfig, Blip2QFormerModel, InstructBlipQFormerModel

from dataclasses import dataclass
from .temporal_spatial_module import TimeConvAdapter

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

logger = logging.get_logger(__name__)


@dataclass
class EncoderBaseModelOutput(BaseModelOutput):
    query_embeds: Optional = None
    sim_t2i: Optional = None



@dataclass
class VideoEncoderOutput(BaseModelOutputWithPastAndCrossAttentions):
    query_embeds: Optional = None


class VideoQformerT5PreTrainedModel(T5PreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)

        # additional re init
        elif isinstance(module, TimeConvAdapter):
            nn.init.constant_(module.fc1.bias, 0.)
            nn.init.constant_(module.fc2.bias, 0.)
            nn.init.constant_(module.conv.weight, 0.)
            nn.init.constant_(module.conv.bias, 0.)



        elif isinstance(
                    module,
                    (T5Model, T5ForConditionalGeneration, T5EncoderModel),
            ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "qa_outputs"):
                module.qa_outputs.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
                module.qa_outputs.bias.data.zero_()
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

        elif isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class VideoT5Encoder(T5Stack):

    def __init__(self, config, embed_tokens=None):
        super().__init__(config=config, embed_tokens=embed_tokens)

        if self.config.use_bbox:
            # ocr_bbox
            self.ocr_bbox_embed = nn.Linear(4, config.d_model)
            self.ocr_bbox_layer_norm = T5LayerNorm(config.d_model)

            if self.config.am_ocr_multi_gran:
                self.line_ocr_bbox_embed = nn.Linear(4, config.d_model)
                self.line_ocr_bbox_layer_norm = T5LayerNorm(config.d_model)

                self.para_ocr_bbox_embed = nn.Linear(4, config.d_model)
                self.para_ocr_bbox_layer_norm = T5LayerNorm(config.d_model)

            self.ocr_bbox_drop = nn.Dropout(config.dropout_rate)




        if self.config.use_frame_type:
            self.frame_type_embedding = nn.Embedding(self.config.num_frames, self.config.d_model)



        # qformer module
        if self.config.use_aggregation_module:
            from .aggregation_module import QFormer
            self.qformer = QFormer(config)

        # attention module
        from transformers.models.bros.modeling_bros import BrosBboxEmbeddings
        bros_config = AutoConfig.from_pretrained(self.config.bbox_module_model_name)


        # bros_config = AutoConfig.from_pretrained("/data/zhangyan/data/pretrained_model/bros-base-uncased")
        self.bros_bbox_embeddings = BrosBboxEmbeddings(bros_config)

        # spatial-temporal module
        from .temporal_spatial_module import VideoT5EncoderLayer
        self.block = nn.ModuleList(
            [VideoT5EncoderLayer(config, num_layer=i, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

            # add
            ocr_bbox=None,
            frame_type=None,
            frcn=None,
            key_frame_module_image_inputs=None,
            key_frame_module_input_ids=None,
            key_frame_module_attention_mask=None,
            qformer_input_ids=None,
            qformer_attention_mask=None,
            itc_targets=None,
            pixel_values=None,

            line_ocr_bbox=None,
            para_ocr_bbox=None,

    ):
        # video
        input_ids = einops.rearrange(input_ids, 'b n l -> (b n) l')
        attention_mask = einops.rearrange(attention_mask, 'b n l -> (b n) l')
        ocr_bbox = einops.rearrange(ocr_bbox, 'b n l d-> (b n) l d')
        frame_type = einops.rearrange(frame_type, 'b n l -> (b n) l')


        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")


        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

            # modify input_embeds
            if self.config.use_bbox:
                ocr_bbox_embeds = self.ocr_bbox_layer_norm(self.ocr_bbox_embed(ocr_bbox.to(torch.float32)))

                if self.config.am_ocr_multi_gran:
                    line_ocr_bbox = einops.rearrange(line_ocr_bbox, 'b n l d-> (b n) l d')
                    para_ocr_bbox = einops.rearrange(para_ocr_bbox, 'b n l d-> (b n) l d')

                    line_ocr_bbox_embeds = self.line_ocr_bbox_layer_norm(self.line_ocr_bbox_embed(line_ocr_bbox.to(torch.float32)))
                    para_ocr_bbox_embeds = self.para_ocr_bbox_layer_norm(self.para_ocr_bbox_embed(para_ocr_bbox.to(torch.float32)))

                    ocr_bbox_embeds[:, :self.config.ocr_len,:] = ocr_bbox_embeds[:, :self.config.ocr_len,:] + line_ocr_bbox_embeds + para_ocr_bbox_embeds

                inputs_embeds = self.ocr_bbox_drop(inputs_embeds + ocr_bbox_embeds)


                if ocr_bbox.shape[-1] == 4:
                    ocr_bbox = ocr_bbox[:, :, [0, 1, 2, 1, 2, 3, 0, 3]]

                scaled_bbox = ocr_bbox * self.config.am_bbox_scale
                bbox_embeds = self.bros_bbox_embeddings(scaled_bbox)

            else: bbox_embeds = None

            if self.config.use_frame_type:
                inputs_embeds += self.frame_type_embedding(frame_type)


        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    bbox_embeds=bbox_embeds, # add
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)


        # convert
        assert hidden_states.shape[1] == self.config.ocr_len + self.config.obj_len
        hidden_states = einops.rearrange(hidden_states, '(b n) l d -> b (n l) d', n=self.config.num_frames)
        attention_mask = einops.rearrange(attention_mask, '(b n) l -> b (n l)',n=self.config.num_frames)


        # add
        if self.config.use_aggregation_module:
            qformer_outputs = self.qformer(encoder_hidden_states=hidden_states,
                                            encoder_attention_mask=attention_mask,
                                            qformer_input_ids=qformer_input_ids,
                                            qformer_attention_mask=qformer_attention_mask,
                                            pixel_values=pixel_values
            )
            query_embeds = qformer_outputs[0]


        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return VideoEncoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            # add
            query_embeds=query_embeds if self.config.use_aggregation_module else None,
        )



class TEA_model(T5ForConditionalGeneration, VideoQformerT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight",
                          "lm_head.weight", "multimodal_embed.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)

        # modify
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = VideoT5Encoder(encoder_config, self.shared)



    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,

            # add
            ocr_bbox: Optional[torch.FloatTensor] = None,
            frame_type: Optional[torch.FloatTensor] = None,
            frcn: Optional[torch.FloatTensor] = None,

            key_frame_module_image_inputs=None,
            key_frame_module_input_ids=None,
            key_frame_module_attention_mask=None,

            qformer_input_ids=None,
            qformer_attention_mask=None,
            itc_targets=None,
            pixel_values=None,

            line_ocr_bbox=None,
            para_ocr_bbox=None,

            **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)


        if encoder_outputs is None:
            # modify
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,

                # add
                ocr_bbox=ocr_bbox,
                frame_type=frame_type,
                frcn=frcn,
                key_frame_module_image_inputs=key_frame_module_image_inputs,
                key_frame_module_input_ids=key_frame_module_input_ids,
                key_frame_module_attention_mask=key_frame_module_attention_mask,

                qformer_input_ids=qformer_input_ids,
                qformer_attention_mask=qformer_attention_mask,
                itc_targets=itc_targets,
                pixel_values=pixel_values,
                line_ocr_bbox=line_ocr_bbox,
                para_ocr_bbox=line_ocr_bbox,

            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            # inference
            encoder_outputs = EncoderBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                # add
                query_embeds=encoder_outputs["query_embeds"] if self.config.use_aggregation_module else None,
            )

        hidden_states = encoder_outputs[0]


        if self.config.use_aggregation_module:
            query_embeds = encoder_outputs["query_embeds"]
            query_attention_mask = torch.ones(query_embeds.size()[:-1], dtype=torch.long, device=attention_mask.device)

            attention_mask = einops.rearrange(attention_mask, 'b n l-> b (n l)', n=self.config.num_frames)
            attention_mask = torch.cat([query_attention_mask, attention_mask], dim=1)
            hidden_states = torch.cat([query_embeds, hidden_states], dim=1)
        else:
            attention_mask = einops.rearrange(attention_mask, 'b n l-> b (n l)', n=self.config.num_frames)


        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # 针对无decoder_input_ids 和 decoder_inputs_embeds的情况，通过labels推断decoder_input_ids
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))


        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }



