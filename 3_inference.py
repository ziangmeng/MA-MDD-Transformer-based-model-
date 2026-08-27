import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import os
from model_options import TransformerClassifier, load_compatible_state_dict

# -------------------- Argument Parser --------------------
parser = argparse.ArgumentParser(description='Run inference using the fine-tuned Transformer model')

parser.add_argument('--inference_data_path', type=str, default='data/Inference_Data_MA.txt', help='Path to input inference data')
parser.add_argument('--model_path', type=str, default='model/MA_transform_learning.pth', help='Path to the fine-tuned model')
parser.add_argument('--output_path', type=str, default='Predicted_SNPs.txt', help='Path to save predicted SNPs')
parser.add_argument('--threshold', type=float, default=0.999, help='Prediction threshold for binary classification')
parser.add_argument('--num_heads', type=int, default=4, help='Number of Transformer attention heads (must match training)')
parser.add_argument('--num_layers', type=int, default=2, help='Number of Transformer encoder layers (must match training)')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of Transformer (must match training)')
parser.add_argument('--use-feature-attention', action='store_true', default=False,
                    help='Use a checkpoint trained with the optional feature-token attention branch')
parser.add_argument('--use-positional-encoding', action='store_true', default=False,
                    help='Add fixed positional encoding to feature tokens')

args = parser.parse_args()
if args.use_positional_encoding and not args.use_feature_attention:
    parser.error('--use-positional-encoding requires --use-feature-attention')

# -------------------- Load Inference Data --------------------
data = pd.read_csv(args.inference_data_path, sep='\t')

inference_features = data[['chr', 'bpos', '2020_beta', '2020_se', '2020_pval', '2020_n',
                           'sqtl', 'sc-eqtl', 'brain_eQTL', 'all_eQTL', 'mQTL',
                           'OCRs_brain', 'OCRs_adult', 'footprints', 'encode',
                           'targetScanS.wgRna', 'tfbsConsSites', 'genomicSuperDups',
                           'reported in previous GWAS', 'ldscore', 'freq', 'ALL_pred']]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(inference_features)
X_tensor = torch.tensor(features_scaled, dtype=torch.float32)

# -------------------- Load Model --------------------
input_dim = X_tensor.shape[1]
model = TransformerClassifier(input_dim=input_dim,
                              num_heads=args.num_heads,
                              num_layers=args.num_layers,
                              hidden_dim=args.hidden_dim,
                              use_feature_attention=args.use_feature_attention,
                              use_positional_encoding=args.use_positional_encoding)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_compatible_state_dict(model, args.model_path, map_location=device)
model = model.to(device)
model.eval()

X_tensor = X_tensor.to(device)

# -------------------- Inference --------------------
with torch.no_grad():
    outputs = model(X_tensor.unsqueeze(1))
    predictions = (outputs > args.threshold).float().cpu().numpy()

# -------------------- Save Predictions --------------------
predicted_snps = data.loc[predictions.flatten() == 1, ['snp', 'chr', 'bpos', '2020_pval']]
predicted_snps.rename(columns={'2020_pval': 'pval'}, inplace=True)

os.makedirs(os.path.dirname(args.output_path), exist_ok=True) if os.path.dirname(args.output_path) else None
predicted_snps.to_csv(args.output_path, sep='\t', index=False)
print(f"Inference results saved to {args.output_path}")
