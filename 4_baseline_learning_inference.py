import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import argparse
import os

# ============ Argument Parser ============
parser = argparse.ArgumentParser(description='Train baseline Transformer model and perform inference (no pretraining)')

parser.add_argument('--train_data_path', type=str, default='data/Train_Data_MA.txt', help='Path to MA training data')
parser.add_argument('--inference_data_path', type=str, default='data/Inference_Data_MA.txt', help='Path to MA inference data')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--num_layers', type=int, default=2, help='Number of Transformer encoder layers')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of feedforward layer')
parser.add_argument('--threshold', type=float, default=0.999, help='Threshold for inference classification')
parser.add_argument('--model_path', type=str, default='model/MA_baseline.pth', help='Path to save trained baseline model')
parser.add_argument('--output_file', type=str, default='Predicted_SNPs_Baseline.txt', help='Path to save inference results')

args = parser.parse_args()

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
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[0, :, :]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ Load and Prepare Training Data ============
df = pd.read_csv(args.train_data_path, sep='\t')

feature_cols = ['chr', 'bpos', '2016_beta', '2016_se', '2016_pval', '2016_n',
                'sqtl', 'sc-eqtl', 'brain_eQTL', 'all_eQTL', 'mQTL',
                'OCRs_brain', 'OCRs_adult', 'footprints', 'encode',
                'targetScanS.wgRna', 'tfbsConsSites', 'genomicSuperDups',
                'reported in previous GWAS', 'ldscore', 'freq', 'ALL_pred']

X = df[feature_cols]
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# ============ Initialize and Train Model ============
model = TransformerClassifier(input_dim=X.shape[1],
                              num_heads=args.num_heads,
                              num_layers=args.num_layers,
                              hidden_dim=args.hidden_dim).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

print("Training baseline model (no pretraining)...")
for epoch in range(args.num_epochs):
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
    print(f"Epoch {epoch+1}/{args.num_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Accuracy: {accuracy:.2f}%")

# ============ Save the Model ============
os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
torch.save(model.state_dict(), args.model_path)
print(f"Baseline model saved to: {args.model_path}")

# ============ Load Inference Data and Predict ============
print("Running inference...")
df_inf = pd.read_csv(args.inference_data_path, sep='\t')
inference_features = df_inf[['chr', 'bpos', '2020_beta', '2020_se', '2020_pval', '2020_n',
                             'sqtl', 'sc-eqtl', 'brain_eQTL', 'all_eQTL', 'mQTL',
                             'OCRs_brain', 'OCRs_adult', 'footprints', 'encode',
                             'targetScanS.wgRna', 'tfbsConsSites', 'genomicSuperDups',
                             'reported in previous GWAS', 'ldscore', 'freq', 'ALL_pred']]

scaler_inf = StandardScaler()
X_inf_scaled = scaler_inf.fit_transform(inference_features)
X_inf_tensor = torch.tensor(X_inf_scaled, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    outputs = model(X_inf_tensor.unsqueeze(1))
    predictions = (outputs > args.threshold).float().cpu().numpy()

predicted_snps = df_inf.loc[predictions.flatten() == 1, ['snp', 'chr', 'bpos', '2020_pval']]
predicted_snps.rename(columns={'2020_pval': 'pval'}, inplace=True)

os.makedirs(os.path.dirname(args.output_file), exist_ok=True) if os.path.dirname(args.output_file) else None
predicted_snps.to_csv(args.output_file, sep='\t', index=False)
print(f"Inference results saved to: {args.output_file}")
