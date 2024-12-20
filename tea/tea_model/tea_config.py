from transformers.models.t5.configuration_t5 import T5Config

class TEAConfig(T5Config):
    model_type = "multimodalt5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
            self,
            vocab_size=32128,
            d_model=512,
            d_kv=64,
            d_ff=2048,
            num_layers=6,
            num_decoder_layers=None,
            num_heads=8,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            initializer_factor=1.0,
            feed_forward_proj="relu",
            is_encoder_decoder=True,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=1,

            # normal settings
            use_frame_type=False,
            use_bbox=False,
            initializer_range=0.02,
            num_frames=10,
            ques_len=45,
            ocr_len=350,
            obj_len=60,

            # aggregation module
            qformer_module_model_name=None,
            use_aggregation_module=False,

            # spatial-temporal module
            am_bbox_scale=None,
            am_ocr_multi_gran=None,
            am_adapter_channels=None,
            am_adapter_kernel_size_t=None,
            am_adapter_kernel_size_l=None,
            bbox_module_model_name=None,

            #
            output_attentions=None,

            **kwargs,
    ):
        # normal settings
        self.num_frames = num_frames
        self.ques_len = ques_len
        self.ocr_len = ocr_len
        self.obj_len = obj_len
        self.use_bbox = use_bbox
        self.use_frame_type = use_frame_type
        self.initializer_range = initializer_range

        # aggregation module
        self.use_aggregation_module = use_aggregation_module
        self.qformer_module_model_name = qformer_module_model_name

        # spatial-temporal module
        self.am_bbox_scale = am_bbox_scale
        self.am_ocr_multi_gran = am_ocr_multi_gran
        self.am_adapter_channels = am_adapter_channels
        self.am_adapter_kernel_size_t=am_adapter_kernel_size_t
        self.am_adapter_kernel_size_l=am_adapter_kernel_size_l
        self.bbox_module_model_name=bbox_module_model_name

        # visualize output attention
        self.output_attentions = output_attentions

        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=d_kv,
            d_ff=d_ff,
            num_layers=num_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            dropout_rate=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_factor=initializer_factor,
            feed_forward_proj=feed_forward_proj,
            is_encoder_decoder=is_encoder_decoder,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
