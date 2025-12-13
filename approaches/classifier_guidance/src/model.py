"""
Noise-Aware Latent Classifier Architecture.

A lightweight CNN that classifies noisy latents at different timesteps
to predict emotion labels. Used for classifier guidance in Stable Diffusion.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding module.
    Projects timesteps to a higher-dimensional space for conditioning.
    """
    
    def __init__(self, dim: int = 256, out_dim: int = 512):
        """
        Initialize time embedding.
        
        Args:
            dim: Input dimension for sinusoidal embedding (default: 256)
            out_dim: Output dimension after projection (default: 512)
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.proj = nn.Linear(dim, out_dim)
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: create sinusoidal embeddings and project.
        
        Args:
            timesteps: Tensor of shape [Batch] with timestep values
            
        Returns:
            Tensor of shape [Batch, out_dim] with time embeddings
        """
        timesteps = timesteps.float()
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = self.proj(emb)
        return emb


class EmotionLatentClassifier(nn.Module):
    """
    Noise-Aware Latent Classifier.
    
    A CNN that takes noisy latents and timesteps as input and predicts
    emotion class logits. Designed to work in the VAE latent space (4x64x64).
    """
    
    def __init__(self, num_emotions: int = 8):
        """
        Initialize the classifier.
        
        Args:
            num_emotions: Number of emotion classes (default: 8)
        """
        super().__init__()
        
        # Time embedding module
        self.time_embed = TimeEmbedding(dim=256, out_dim=512)
        
        # Convolutional blocks
        # Block 1: 4 -> 64 channels, stride 1 (64x64 -> 64x64)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Block 2: 64 -> 128 channels, stride 2 (64x64 -> 32x32)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Block 3: 128 -> 256 channels, stride 2 (32x32 -> 16x16)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Block 4: 256 -> 512 channels, stride 2 (16x16 -> 8x8)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Time projection for feature injection
        self.time_proj = nn.Linear(512, 512)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_emotions)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x: Noisy latents of shape [Batch, 4, 64, 64]
            timesteps: Timestep values of shape [Batch]
            
        Returns:
            Logits of shape [Batch, num_emotions]
        """
        # CRITICAL: Ensure inputs are float32 to match model parameters
        x = x.to(dtype=torch.float32)
        timesteps = timesteps.to(dtype=torch.float32)
        
        # Convolutional feature extraction
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))
        x = F.silu(self.bn4(self.conv4(x)))
        
        # Time embedding and injection
        time_emb = self.time_proj(self.time_embed(timesteps))
        time_emb = time_emb[:, :, None, None].expand_as(x)
        x = x + time_emb  # Add time conditioning
        
        # Global pooling and classification
        x = self.global_pool(x).flatten(1)
        logits = self.classifier(x)  # [B, num_emotions]
        
        return logits

