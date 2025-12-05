"""
Classifier Guidance Approach Package.

This package implements classifier guidance for Stable Diffusion v1.5,
using a noise-aware latent classifier to guide image generation.
"""

from .model import EmotionLatentClassifier, TimeEmbedding

__all__ = ['EmotionLatentClassifier', 'TimeEmbedding']

