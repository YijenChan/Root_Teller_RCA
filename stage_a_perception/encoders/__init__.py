# -*- coding: utf-8 -*-
"""
encoders registry for Stage-A perception.
Provides simple modular imports for HashingTextEncoder,
MetricsEncoder, and TraceEncoder.
"""

from .text_encoder import HashingTextEncoder, OpenAIEmbedder
from .metrics_encoder import MetricsEncoder
from .trace_encoder import TraceEncoder

ENCODER_REGISTRY = {
    "log_hash": HashingTextEncoder,
    "log_openai": OpenAIEmbedder,
    "metrics_cnn": MetricsEncoder,
    "trace_gru": TraceEncoder,
}

def get_encoder(name: str):
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder: {name}")
    return ENCODER_REGISTRY[name]
