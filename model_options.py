import math
import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    """Legacy-compatible classifier with optional feature-token attention."""

    def __init__(self, input_dim, num_heads, num_layers, hidden_dim,
                 output_dim=1, use_feature_attention=False,
                 use_positional_encoding=False):
        super().__init__()
        if use_positional_encoding and not use_feature_attention:
            raise ValueError("Positional encoding requires feature attention")
        self.use_feature_attention = use_feature_attention
        self.use_positional_encoding = use_positional_encoding

        # Existing path: names and shapes are kept for checkpoint compatibility.
        self.input_mapping = nn.Linear(input_dim, 20)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=20, nhead=num_heads, dim_feedforward=hidden_dim
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(20, output_dim)
        self.sigmoid = nn.Sigmoid()

        if use_feature_attention:
            self.feature_projection = nn.Linear(1, 20)
            self.feature_embeddings = nn.Parameter(torch.zeros(1, input_dim, 20))
            self.feature_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=20,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                ),
                num_layers=num_layers,
            )
            self.feature_fc = nn.Linear(20, output_dim)
            # Starting at zero preserves the existing prediction path initially.
            self.feature_attention_scale = nn.Parameter(torch.tensor(0.0))
            if use_positional_encoding:
                position = torch.arange(input_dim, dtype=torch.float32).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, 20, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / 20)
                )
                positional_encoding = torch.zeros(1, input_dim, 20)
                positional_encoding[0, :, 0::2] = torch.sin(position * div_term)
                positional_encoding[0, :, 1::2] = torch.cos(position * div_term)
                self.register_buffer(
                    "positional_encoding", positional_encoding, persistent=False
                )

    def forward(self, x):
        raw_features = x.squeeze(1) if x.dim() == 3 and x.size(1) == 1 else x

        legacy = self.input_mapping(x)
        legacy = legacy.permute(1, 0, 2)
        legacy = self.transformer(legacy)
        logits = self.fc(legacy[0, :, :])

        if self.use_feature_attention:
            feature_tokens = self.feature_projection(raw_features.unsqueeze(-1))
            feature_tokens = feature_tokens + self.feature_embeddings
            if self.use_positional_encoding:
                feature_tokens = feature_tokens + self.positional_encoding
            feature_context = self.feature_transformer(feature_tokens).mean(dim=1)
            logits = logits + self.feature_attention_scale * self.feature_fc(feature_context)

        return self.sigmoid(logits)


def load_compatible_state_dict(model, checkpoint_path, map_location=None):
    """Load legacy checkpoints while allowing a newly enabled optional branch."""
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    if not model.use_feature_attention:
        model.load_state_dict(state_dict)
        return

    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith((
            "feature_projection.",
            "feature_embeddings",
            "feature_transformer.",
            "feature_fc.",
            "feature_attention_scale",
        ))
    ]
    if unexpected or invalid_missing:
        raise RuntimeError(
            f"Incompatible checkpoint. Missing keys: {invalid_missing}; "
            f"unexpected keys: {unexpected}"
        )
