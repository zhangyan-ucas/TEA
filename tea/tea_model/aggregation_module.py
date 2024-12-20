from transformers import AutoConfig, Blip2QFormerModel, InstructBlipVisionModel,InstructBlipQFormerModel
import torch.nn as nn
import torch
from typing import Optional, Tuple, Union
import einops
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)

from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from dataclasses import dataclass
from transformers.activations import ACT2FN

@dataclass
class QFormer_Output(BaseModelOutput):
    language_model_inputs: Optional = None


class QFormer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        qformer_module_config = AutoConfig.from_pretrained(config.qformer_module_model_name)

        # modify config
        self.qformer_module_config = qformer_module_config

        # step 1 linear
        self.linear_proj_1 = nn.Linear(config.d_model, qformer_module_config.qformer_config.encoder_hidden_size,bias=True)
        self.act = ACT2FN["gelu"]
        self.linear_proj_2 = nn.Linear(qformer_module_config.qformer_config.encoder_hidden_size,
                                       qformer_module_config.qformer_config.encoder_hidden_size, bias=True)

        # step 2 qformer
        self.query_tokens = nn.Parameter(
            torch.zeros(1, qformer_module_config.num_query_tokens, qformer_module_config.qformer_config.hidden_size))


        self.vision_encoder = InstructBlipVisionModel(self.qformer_module_config.vision_config)
        self.vision_encoder.requires_grad = False

        self.query_d_proj = nn.Linear(qformer_module_config.vision_config.hidden_size, qformer_module_config.qformer_config.hidden_size)
        self.query_act = ACT2FN["gelu"]
        self.query_l_proj = nn.Linear(self.config.num_frames, qformer_module_config.num_query_tokens, bias=False)

        self.qformer = InstructBlipQFormerModel(qformer_module_config.qformer_config)


        # step 3 language project
        self.language_projection = nn.Linear(qformer_module_config.qformer_config.hidden_size, config.d_model)


    def forward(
            self,
            encoder_hidden_states: torch.FloatTensor,
            encoder_attention_mask: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            qformer_input_ids: Optional = None,
            qformer_attention_mask: Optional = None,
            pixel_values: Optional = None,
    ):
        # query encoder hidden states focous all or only focus ocr
        image_embeds = einops.rearrange(encoder_hidden_states, 'b (n l) d -> b n l d', n=self.config.num_frames)[:,
                       :, :self.config.ocr_len, :]
        encoder_attention_mask = einops.rearrange(encoder_attention_mask, 'b (n l) -> b n l',
                                                  n=self.config.num_frames)[:, :, :self.config.ocr_len]

        image_embeds = einops.rearrange(image_embeds, 'b n l d -> b (n l) d', n=self.config.num_frames)
        encoder_attention_mask = einops.rearrange(encoder_attention_mask, 'b n l -> b (n l)',
                                                  n=self.config.num_frames)

        # step 1 proj
        image_embeds = self.linear_proj_1(image_embeds)
        image_embeds = self.act(image_embeds)
        image_embeds = self.linear_proj_2(image_embeds)

        # step 2 forward the query tokens
        if not self.config.qm_global_query_init:
            query_tokens = self.query_tokens.expand(encoder_hidden_states.shape[0], -1, -1)
        else:
            pixel_values = einops.rearrange(pixel_values, 'b n c h w -> (b n) c h w', n=self.config.num_frames)

            with torch.no_grad():
                global_vision_embeds = self.vision_encoder(pixel_values).pooler_output

            global_vision_embeds = self.query_act(self.query_d_proj(global_vision_embeds))
            global_vision_embeds = einops.rearrange(global_vision_embeds, '(b n) d -> b d n', n=self.config.num_frames)
            global_vision_embeds = self.query_l_proj(global_vision_embeds).permute(0, 2, 1)

            query_tokens = self.query_tokens.expand(encoder_hidden_states.shape[0], -1, -1) + global_vision_embeds


        # attention mask
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)

        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3
        language_model_inputs = self.language_projection(query_output)




        output = QFormer_Output(
            language_model_inputs=language_model_inputs,
        )

        return output