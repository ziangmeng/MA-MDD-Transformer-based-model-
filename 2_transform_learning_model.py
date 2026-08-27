import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import argparse
import os
from model_options import TransformerClassifier, load_compatible_state_dict

# -------------------- Argument Parser --------------------
parser = argparse.ArgumentParser(description='Transfer learning using pre-trained Transformer model')

parser.add_argument('--data_path', type=str, default='data/Train_Data_MA.txt', help='Path to MA training data')
parser.add_argument('--pretrained_model_path', type=str, default='model/MDD_model.pth', help='Path to pre-trained MDD model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for fine-tuning')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--num_layers', type=int, default=2, help='Number of Transformer layers')
parser.add_argument('--hidden_dim', type=int, default=64, help='Feedforward hidden dimension in Transformer')
parser.add_argument('--output_path', type=str, default='model/MA_transform_learning.pth', help='Path to save fine-tuned model')
transfer_group = parser.add_mutually_exclusive_group()
transfer_group.add_argument('--use-transfer-learning', dest='use_transfer_learning', action='store_true',
                            help='Initialize from the pretrained checkpoint')
transfer_group.add_argument('--no-use-transfer-learning', dest='use_transfer_learning', action='store_false',
                            help='Train on the target data without loading pretrained weights')
parser.set_defaults(use_transfer_learning=True)
parser.add_argument('--use-feature-attention', action='store_true', default=False,
                    help='Enable the optional feature-token attention branch')
parser.add_argument('--use-positional-encoding', action='store_true', default=False,
                    help='Add fixed positional encoding to feature tokens')

args = parser.parse_args()
if args.use_positional_encoding and not args.use_feature_attention:
    parser.error('--use-positional-encoding requires --use-feature-attention')

# -------------------- Load and Preprocess Data --------------------
df = pd.read_csv(args.data_path, sep='\t')

feature_cols = ['chr', 'bpos', '2016_beta', '2016_se', '2016_pval', '2016_n',
                'sqtl', 'sc-eqtl', 'brain_eQTL', 'all_eQTL', 'mQTL',
                'OCRs_brain', 'OCRs_adult', 'footprints', 'encode',
                'targetScanS.wgRna', 'tfbsConsSites', 'genomicSuperDups',
                'reported in previous GWAS', 'ldscore', 'freq', 'ALL_pred']

features = df[feature_cols]
labels = df['label']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_tensor = torch.tensor(features_scaled, dtype=torch.float32)
y_tensor = torch.tensor(labels.values, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# -------------------- Load Pretrained Model --------------------
model = TransformerClassifier(
    input_dim=features.shape[1],
    num_heads=args.num_heads,
    num_layers=args.num_layers,
    hidden_dim=args.hidden_dim,
    use_feature_attention=args.use_feature_attention,
    use_positional_encoding=args.use_positional_encoding,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.use_transfer_learning:
    load_compatible_state_dict(model, args.pretrained_model_path, map_location=device)
    print(f"Initialized from pretrained model: {args.pretrained_model_path}")
else:
    print("Training on target data without pretrained initialization")
model = model.to(device)

# -------------------- Training --------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch.unsqueeze(1))
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val.unsqueeze(1))
            loss = criterion(outputs, y_val)
            val_loss += loss.item()
            predicted = (outputs > 0.9).float()
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {accuracy:.2f}%")

# -------------------- Final Evaluation --------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_val, y_val in val_loader:
        X_val, y_val = X_val.to(device), y_val.to(device)
        outputs = model(X_val.unsqueeze(1))
        predicted = (outputs > 0.9).float()
        total += y_val.size(0)
        correct += (predicted == y_val).sum().item()

accuracy = 100 * correct / total
print(f"Final Accuracy on Validation Set: {accuracy:.2f}%")

# -------------------- Save Model --------------------
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
torch.save(model.state_dict(), args.output_path)
print(f"Model saved to {args.output_path}")
