import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os

# ============ Define Transformer Classifier ============
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim=1):
        super(TransformerClassifier, self).__init__()
        self.input_mapping = nn.Linear(input_dim, 20)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=20, nhead=num_heads, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.fc = nn.Linear(20, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_mapping(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, feature_dim]
        x = self.transformer(x)
        x = x[0, :, :]  # Use the first token's output
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# ============ Load and Prepare Training Data ============
train_data_path = 'data/Train_Data_MA.txt'
df = pd.read_csv(train_data_path, sep='\t')

feature_cols = ['chr', 'bpos', '2016_beta', '2016_se', '2016_pval', '2016_n',
                'sqtl', 'sc-eqtl', 'brain_eQTL', 'all_eQTL', 'mQTL',
                'OCRs_brain', 'OCRs_adult', 'footprints', 'encode',
                'targetScanS.wgRna', 'tfbsConsSites', 'genomicSuperDups',
                'reported in previous GWAS', 'ldscore', 'freq', 'ALL_pred']
label_col = 'label'

X = df[feature_cols]
y = df[label_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ Initialize Baseline Model (no pretraining) ============
model = TransformerClassifier(input_dim=X.shape[1], num_heads=4, num_layers=2, hidden_dim=64).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ============ Train the Baseline Model ============
print("Training baseline model (no pretraining)...")
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch.unsqueeze(1))
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val.unsqueeze(1))
            predicted = (outputs > 0.9).float()
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Accuracy: {accuracy:.2f}%")

# ============ Save the Trained Baseline Model ============
os.makedirs('model', exist_ok=True)
model_path = 'model/MA_baseline.pth'
torch.save(model.state_dict(), model_path)
print(f"Baseline model saved to: {model_path}")

# ============ Load Inference Data and Run Predictions ============
print("Running inference...")

inference_path = 'data/Inference_Data_MA.txt'
df_inf = pd.read_csv(inference_path, sep='\t')

inference_features = df_inf[['chr', 'bpos', '2020_beta', '2020_se', '2020_pval', '2020_n',
                             'sqtl', 'sc-eqtl', 'brain_eQTL', 'all_eQTL', 'mQTL',
                             'OCRs_brain', 'OCRs_adult', 'footprints', 'encode',
                             'targetScanS.wgRna', 'tfbsConsSites', 'genomicSuperDups',
                             'reported in previous GWAS', 'ldscore', 'freq', 'ALL_pred']]

# Normalize inference features (assuming independent standardization)
scaler_inf = StandardScaler()
X_inf_scaled = scaler_inf.fit_transform(inference_features)
X_inf_tensor = torch.tensor(X_inf_scaled, dtype=torch.float32).to(device)

# Load the trained model and predict
model.eval()
with torch.no_grad():
    outputs = model(X_inf_tensor.unsqueeze(1))
    predictions = (outputs > 0.999).float().cpu().numpy()

# Extract predicted SNPs
predicted_snps = df_inf.loc[predictions.flatten() == 1, ['snp', 'chr', 'bpos', '2020_pval']]
predicted_snps.rename(columns={'2020_pval': 'pval'}, inplace=True)

# Save prediction results
output_file = 'Predicted_SNPs_Baseline.txt'
predicted_snps.to_csv(output_file, sep='\t', index=False)
print(f"Inference results saved to: {output_file}")
