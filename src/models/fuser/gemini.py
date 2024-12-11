from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Fuser


class GeminiSinglePair(nn.Module):
    def __init__(self, d_model=768, num_heads=8, mlp_hidden_dim=512):
        super(GeminiSinglePair, self).__init__()

        # Define shared layers for Q, K, V
        self.qkv = nn.Linear(d_model, 3 * d_model)  # Shared Q, K, V projection

        # MLP for combining k_img and k_depth
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, mlp_hidden_dim),  # First layer
            nn.ReLU(),  # Activation
            nn.Linear(mlp_hidden_dim, d_model),  # Output to match key dimension
        )

        # Multi-head attention layers
        self.attn = nn.MultiheadAttention(d_model, num_heads)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, image_feature: torch.Tensor, auxiliary_feature: torch.Tensor
    ) -> torch.Tensor:
        # batch_size = image_tokens.size(0)

        # Concatenate image and depth tokens (along sequence dimension)
        combined_feature = torch.cat(
            [image_feature, auxiliary_feature], dim=1
        )  # Shape: [batch_size, 2 * num_tokens, d_model]

        # Apply shared Q, K, V projection
        qkv = self.qkv(
            combined_feature
        )  # Shape: [batch_size, 2 * num_tokens, 3 * d_model]
        q, k, v = qkv.chunk(3, dim=-1)  # Split into Q, K, V

        # Separate the Q, K, V for image and depth tokens
        q_img = q[:, : image_feature.size(1)]  # Image query tokens
        k_img = k[:, : image_feature.size(1)]  # Image key tokens
        v_img = v[:, : image_feature.size(1)]  # Image value tokens

        q_depth = q[:, image_feature.size(1) :]  # Depth query tokens
        k_depth = k[:, image_feature.size(1) :]  # Depth key tokens
        v_depth = v[:, image_feature.size(1) :]  # Depth value tokens

        # Apply the MLP + Softmax to combine K_image and K_depth
        k_combined_img = self.mlp(
            torch.cat([k_img, k_depth], dim=-1)
        )  # Combine keys using MLP
        k_combined_img = F.softmax(k_combined_img, dim=-1)  # Apply softmax

        k_combined_depth = self.mlp(
            torch.cat([k_depth, k_img], dim=-1)
        )  # Combine keys using MLP
        k_combined_depth = F.softmax(k_combined_depth, dim=-1)  # Apply softmax

        # Transpose for multihead attention (Shape: [seq_len, batch_size, d_model])
        # q_img = q_img.transpose(0, 1)
        # k_img = k_img.transpose(0, 1)
        # v_img = v_img.transpose(0, 1)

        # q_depth = q_depth.transpose(0, 1)
        # k_depth = k_depth.transpose(0, 1)
        # v_depth = v_depth.transpose(0, 1)

        # First embedding: Attention on Image tokens + Joint attention (Image + Depth)
        attn_img, _ = self.attn(q_img, k_img, v_img)  # Attention using image tokens
        attn_joint_img_depth, _ = self.attn(
            q_img, k_combined_img, v_depth
        )  # Joint attention using MLP combined keys

        # Second embedding: Attention on Depth tokens + Joint attention (Image + Depth)
        attn_depth, _ = self.attn(
            q_depth, k_depth, v_depth
        )  # Attention using depth tokens
        attn_joint_depth_img, _ = self.attn(
            q_depth, k_combined_depth, v_img
        )  # Joint attention using MLP combined keys

        # Add the attention outputs
        embedding_1 = attn_img + attn_joint_img_depth
        embedding_2 = attn_depth + attn_joint_depth_img

        # Add embeddings and apply LayerNorm
        combined_embedding = embedding_1 + embedding_2
        combined_embedding = self.norm(combined_embedding)

        return combined_embedding


class GeminiFuser(Fuser):
    # TODO: Use model params configs
    def __init__(
        self, n_auxiliary_features: int, d_model=768, num_heads=8, mlp_hidden_dim=512
    ):
        super().__init__(n_auxiliary_features=n_auxiliary_features)

        self.fusers = nn.ModuleList(
            [
                GeminiSinglePair(
                    d_model=d_model, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim
                ).bfloat16()
                for _ in range(self.n_auxiliary_features)
            ]
        )

    def forward(
        self, image_feature: torch.Tensor, auxiliary_features: List[torch.Tensor]
    ) -> torch.Tensor:
        fused_features = []
        for i, fuser in enumerate(self.fusers):
            fused_features.append(fuser(image_feature, auxiliary_features[i]))

        fusion = 0
        for fused_feature in fused_features:
            fusion += fused_feature

        return fusion / self.n_auxiliary_features
