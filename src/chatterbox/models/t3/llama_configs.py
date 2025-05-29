# FILE: src/chatterbox/models/t3/llama_configs.py
LLAMA_520M_CONFIG_DICT = dict(
    # Arbitrary small number that won't cause problems when loading.
    # These param are unused due to custom input layers.
    vocab_size=8,
    # This defines the maximum sequence length the RoPE mechanism is initially configured for
    # if rope_scaling is None. 8192 is a common value for Llama models.
    max_position_embeddings=8192,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=30,
    num_attention_heads=16,
    attn_implementation="sdpa", # Use "eager" if "sdpa" causes issues with older torch/transformers
    head_dim=64, # hidden_size // num_attention_heads
    tie_word_embeddings=False,
    hidden_act="silu",
    attention_bias=False,
    attention_dropout=0.0,
    initializer_range=0.02,
    mlp_bias=False,
    model_type="llama",
    num_key_value_heads=16, # For Llama GQA/MQA. For standard MHA, num_key_value_heads = num_attention_heads
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=None, # MODIFICATION: Explicitly set to None
    rope_theta=500000.0, # This is crucial for Llama 3 style RoPE
    torch_dtype="bfloat16", # Consider "float16" or "float32" if bf16 is not supported or causes issues
    use_cache=True,
)

LLAMA_CONFIGS = {
    "Llama_520M": LLAMA_520M_CONFIG_DICT,
}