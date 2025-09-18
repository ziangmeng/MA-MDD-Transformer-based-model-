import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, hidden_dim=64, output_dim=1, p_drop=0.05):
        super().__init__()
        self.input_mapping = nn.Linear(input_dim, 20)
        self.input_dropout = nn.Dropout(p_drop)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=20, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=0.1, activation='relu', batch_first=False
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(20, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):               # x: [B, D]
        x = self.input_mapping(x)
        x = self.input_dropout(x)
        x = x.unsqueeze(1).permute(1, 0, 2)  # [1, B, 20]
        x = self.transformer(x)[0]           # [B, 20]
        x = self.fc(x)                       # [B, 1]
        return self.sigmoid(x)
