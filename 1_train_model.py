import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import argparse
import os

# -------------------- Argument Parser --------------------
parser = argparse.ArgumentParser(description='Train Transformer on MDD dataset')

parser.add_argument('--data_path', type=str, default='data/Train_Data_MDD.txt', help='Path to training data')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--num_layers', type=int, default=2, help='Number of Transformer layers')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of feedforward layer')
parser.add_argument('--output_path', type=str, default='model/MDD_model.pth', help='Path to save model')

args = parser.parse_args()

# -------------------- Load and Preprocess Data --------------------
final_training_set_mdd = pd.read_csv(args.data_path, sep='\t')

feature_cols = ['chr', 'bpos', '2018_mdd_beta', '2018_mdd_se', '2018_mdd_pval',
                '2018_mdd_n', 'sqtl', 'sc-eqtl', 'brain_eQTL', 'all_eQTL', 'mQTL',
                'OCRs_brain', 'OCRs_adult', 'footprints', 'encode',
                'targetScanS.wgRna', 'tfbsConsSites', 'genomicSuperDups',
                'reported in previous GWAS', 'ldscore', 'freq', 'ALL_pred']

features_mdd = final_training_set_mdd[feature_cols]
labels_mdd = final_training_set_mdd['label']

scaler_mdd = StandardScaler()
features_scaled_mdd = scaler_mdd.fit_transform(features_mdd)

X_tensor_mdd = torch.tensor(features_scaled_mdd, dtype=torch.float32)
y_tensor_mdd = torch.tensor(labels_mdd.values, dtype=torch.float32).view(-1, 1)

dataset_mdd = TensorDataset(X_tensor_mdd, y_tensor_mdd)
train_size_mdd = int(0.8 * len(dataset_mdd))
val_size_mdd = len(dataset_mdd) - train_size_mdd
train_dataset_mdd, val_dataset_mdd = random_split(dataset_mdd, [train_size_mdd, val_size_mdd])

train_loader_mdd = DataLoader(train_dataset_mdd, batch_size=args.batch_size, shuffle=True)
val_loader_mdd = DataLoader(val_dataset_mdd, batch_size=args.batch_size, shuffle=False)

# -------------------- Define Model --------------------
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

input_dim_mdd = features_scaled_mdd.shape[1]
model = TransformerClassifier(input_dim=input_dim_mdd,
                              num_heads=args.num_heads,
                              num_layers=args.num_layers,
                              hidden_dim=args.hidden_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# -------------------- Training Loop --------------------
for epoch in range(args.num_epochs):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader_mdd:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch.unsqueeze(1))
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_val, y_val in val_loader_mdd:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val.unsqueeze(1))
            loss = criterion(outputs, y_val)
            val_loss += loss.item()
            predicted = (outputs > 0.9).float()
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {running_loss/len(train_loader_mdd):.4f}, "
          f"Val Loss: {val_loss/len(val_loader_mdd):.4f}, Accuracy: {accuracy:.2f}%")

# -------------------- Save Model --------------------
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
torch.save(model.state_dict(), args.output_path)
print(f"Model saved to {args.output_path}")
