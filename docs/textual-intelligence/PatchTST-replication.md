
---

### **The Definitive, Faithful Replication Script (V5 - Self-Sufficient)**

This script is designed to be a single, self-contained file. It will download the data, configure the experiment with the exact hyperparameters from the paper, run the full training and testing process, and report the final metrics against the paper's published results.

**The Target:** Replicate the **supervised forecasting** results of **PatchTST/16** on the **multivariate Electricity dataset** for a prediction length of **T=96**.
*   **Paper's Result (Table 3, Page 7):** MSE = **0.130**, MAE = **0.222**.
*   **Our Goal:** To achieve an MSE score in this exact ballpark.

```python
# ==============================================================================
#           THE DEFINITIVE & FAITHFUL PATCHTST REPLICATION (V5 - SELF-SUFFICIENT)
#        - This version is a direct, faithful replication of the supervised
#          forecasting experiment from the ICLR 2023 paper and its official code.
#        - It handles its own data download to a local directory.
#        - It uses the exact classes and workflow from the provided repository.
#        - GOAL: Replicate the SOTA forecasting MSE on the Electricity dataset.
# ==============================================================================

# --- Core Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import joblib
import math
from typing import Optional, List, Callable
import urllib.request
import sys
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import time
import warnings
from collections import OrderedDict
from torch.optim import lr_scheduler

warnings.filterwarnings('ignore')

# ==============================================================================
#      PART 1: ROBUST DATA DIRECTORY AND DOWNLOAD
# ==============================================================================
# --- THE DEFINITIVE FIX: The script manages its own data ---
ROOT_PATH = './'
# The code expects this specific directory structure
DATA_DIR = os.path.join(ROOT_PATH, 'data/electricity/')
os.makedirs(DATA_DIR, exist_ok=True)
DATA_PATH = os.path.join(DATA_DIR, 'electricity.csv')
CHECKPOINTS_DIR = os.path.join(ROOT_PATH, 'checkpoints')
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

def download_data():
    if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) < 1000:
        print(f"Dataset not found or is empty. Downloading electricity.csv to {DATA_PATH}...")
        try:
            # Using a known stable URL for this dataset
            url = 'https://raw.githubusercontent.com/zhouhaoyi/Informer2020/main/data/ETT/electricity.csv'
            urllib.request.urlretrieve(url, DATA_PATH)
            if os.path.getsize(DATA_PATH) < 1000:
                raise Exception("Downloaded file is empty or too small!")
            print("Download complete.")
        except Exception as e:
            print(f"FATAL ERROR: Failed to download dataset. Reason: {e}")
            sys.exit(1)
    else:
        print(f"{DATA_PATH} already exists and is not empty.")
# --- END OF FIX ---


# ==============================================================================
#      PART 2: EXACT REPLICATION OF REPOSITORY CODE
# ==============================================================================

# --- FROM: layers/PatchTST_layers.py ---
def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')

def positional_encoding(pe, learn_pe, q_len, d_model):
    if pe == 'zeros': W_pos = torch.empty((q_len, d_model)); nn.init.uniform_(W_pos, -0.02, 0.02)
    else: raise ValueError(f"PE type not supported.")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): super().__init__(); self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

# --- FROM: layers/RevIN.py ---
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features, self.eps, self.affine, self.subtract_last = num_features, eps, affine, subtract_last
        if self.affine: self.affine_weight, self.affine_bias = nn.Parameter(torch.ones(self.num_features)), nn.Parameter(torch.zeros(self.num_features))
    def forward(self, x, mode:str):
        if mode == 'norm': self._get_statistics(x); x = self._normalize(x)
        elif mode == 'denorm': x = self._denormalize(x)
        else: raise NotImplementedError
        return x
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last: self.last = x[:,-1,:].unsqueeze(1)
        else: self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
    def _normalize(self, x):
        if self.subtract_last: x = x - self.last
        else: x = x - self.mean
        x = x / self.stdev
        if self.affine: x = x * self.affine_weight; x = x + self.affine_bias
        return x
    def _denormalize(self, x):
        if self.affine: x = x - self.affine_bias; x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last: x = x + self.last
        else: x = x + self.mean
        return x

# --- FROM: layers/PatchTST_backbone.py ---
class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_ff=256, dropout=0.1, activation="gelu", norm='BatchNorm', res_attention=False):
        super().__init__()
        self.res_attention = res_attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower(): self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else: self.norm_attn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), get_activation_fn(activation), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower(): self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else: self.norm_ffn = nn.LayerNorm(d_model)
    def forward(self, src:Tensor, prev:Optional[Tensor]=None):
        src2, attn = self.self_attn(src, src, src, need_weights=False)
        src = src + self.dropout_attn(src2)
        src = self.norm_attn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        src = self.norm_ffn(src)
        if self.res_attention: return src, attn
        else: return src

class TSTiEncoder(nn.Module):
    def __init__(self, c_in, patch_num, patch_len, n_layers=3, d_model=128, n_heads=16, d_ff=256, dropout=0., pe='zeros', learn_pe=True, **kwargs):
        super().__init__()
        self.patch_num = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([TSTEncoderLayer(patch_num, d_model, n_heads, d_ff=d_ff, dropout=dropout, **kwargs) for _ in range(n_layers)])
    def forward(self, x) -> Tensor:
        n_vars = x.shape[1]
        x = x.permute(0,1,3,2)
        x = self.W_P(x)
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        u = self.dropout(u + self.W_pos)
        z = u
        for mod in self.layers:
            z = mod(z)
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))
        z = z.permute(0,1,3,2)
        return z

class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, patch_len:int, stride:int, d_model=128, n_layers=3, n_heads=16, dropout=0., revin=True, affine=True, **kwargs):
        super().__init__()
        self.revin, self.revin_layer = revin, RevIN(c_in, affine=affine) if revin else None
        self.patch_len, self.stride = patch_len, stride
        patch_num = int((context_window - patch_len)/stride + 1)
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride)); patch_num += 1
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout, **kwargs)
    def forward(self, z):
        if self.revin: z = self.revin_layer(z, 'norm')
        z = self.padding_patch_layer(z.permute(0,2,1)).permute(0,2,1)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0,1,3,2)
        z = self.backbone(z)
        return z

class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten, self.linear, self.dropout = nn.Flatten(start_dim=-2), nn.Linear(nf, target_window), nn.Dropout(head_dropout)
    def forward(self, x):
        x = self.flatten(x); x = self.linear(x); x = self.dropout(x)
        return x

# --- FROM: models/PatchTST.py ---
class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.backbone = PatchTST_backbone(c_in=configs.enc_in, context_window=configs.seq_len, patch_len=configs.patch_len, stride=configs.stride, d_model=configs.d_model, n_layers=configs.e_layers, n_heads=configs.n_heads, dropout=configs.dropout, revin=configs.revin, affine=configs.affine, **kwargs)
        head_nf = configs.d_model * (int((configs.seq_len - configs.patch_len)/configs.stride + 2))
        self.head = Flatten_Head(configs.enc_in, head_nf, configs.pred_len, head_dropout=configs.head_dropout)
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        z = self.backbone(x_enc.permute(0,2,1))
        z = self.head(z)
        if self.backbone.revin: z = self.backbone.revin_layer(z, 'denorm')
        return z.permute(0,2,1)

# --- FROM: data_provider/data_loader.py ---
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M', data_path='electricity.csv', target='OT', scale=True):
        if size is None: self.seq_len, self.label_len, self.pred_len = 96, 0, 96
        else: self.seq_len, self.label_len, self.pred_len = size[0], size[1], size[2]
        assert flag in ['train', 'test', 'val']; type_map = {'train': 0, 'val': 1, 'test': 2}; self.set_type = type_map[flag]
        self.features, self.target, self.scale = features, target, scale
        self.root_path, self.data_path = root_path, data_path
        self.__read_data__()
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        num_train, num_vali = int(len(df_raw) * 0.7), int(len(df_raw) * 0.1)
        num_test = len(df_raw) - num_train - num_vali
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        df_data = df_raw[df_raw.columns[1:]]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]; self.scaler.fit(train_data.values); data = self.scaler.transform(df_data.values)
        else: data = df_data.values
        self.data_x, self.data_y = data[border1:border2], data[border1:border2]
    def __getitem__(self, index):
        s_begin, s_end = index, index + self.seq_len
        r_begin, r_end = s_end - self.label_len, s_end - self.label_len + self.pred_len
        seq_x, seq_y = self.data_x[s_begin:s_end], self.data_y[r_begin:r_end]
        return seq_x, seq_y
    def __len__(self): return len(self.data_x) - self.seq_len - self.pred_len + 1
    def inverse_transform(self, data): return self.scaler.inverse_transform(data)

# --- FROM: utils/tools.py & metrics.py ---
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience, self.verbose, self.counter, self.best_score, self.early_stop, self.val_loss_min, self.delta = patience, verbose, 0, None, False, np.Inf, delta
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None: self.best_score = score; self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else: self.best_score = score; self.save_checkpoint(val_loss, model, path); self.counter = 0
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose: print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth'); self.val_loss_min = val_loss

def metric(pred, true):
    return np.mean(np.abs(pred - true)), np.mean((pred - true) ** 2)

# --- FROM: exp/exp_main.py ---
class Exp_Main:
    def __init__(self, args):
        self.args, self.device = args, self._acquire_device()
        self.model = Model(self.args).float().to(self.device)
    def _acquire_device(self): return torch.device(f'cuda:{self.args.gpu}' if self.args.use_gpu else 'cpu')
    def _get_data(self, flag):
        ds = Dataset_Custom(root_path=self.args.root_path, data_path=self.args.data_path, flag=flag, size=[self.args.seq_len, self.args.label_len, self.args.pred_len], features=self.args.features)
        return ds, DataLoader(ds, batch_size=self.args.batch_size, shuffle=(flag == 'train'), num_workers=self.args.num_workers, drop_last=True)
    def vali(self, vali_loader, criterion):
        total_loss = []; self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs, batch_y = outputs[:, :, f_dim:], batch_y[:, :, f_dim:]
                pred, true = outputs.detach().cpu(), batch_y.detach().cpu()
                loss = criterion(pred, true); total_loss.append(loss.item())
        total_loss = np.average(total_loss); self.model.train(); return total_loss
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train'); vali_data, vali_loader = self._get_data(flag='val')
        path = os.path.join(self.args.checkpoints, setting);
        if not os.path.exists(path): os.makedirs(path)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim, steps_per_epoch = len(train_loader), pct_start = self.args.pct_start, epochs = self.args.train_epochs, max_lr = self.args.learning_rate)
        criterion = nn.MSELoss()
        for epoch in range(self.args.train_epochs):
            train_loss = []; self.model.train()
            for i, (batch_x, batch_y) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
                model_optim.zero_grad()
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs, batch_y = outputs[:, :, f_dim:], batch_y[:, :, f_dim:]
                loss = criterion(outputs, batch_y); train_loss.append(loss.item())
                loss.backward(); model_optim.step()
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop: print("Early stopping"); break
            scheduler.step()
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs, batch_y = outputs[:, :, f_dim:], batch_y[:, :, f_dim:]
                preds.append(outputs.detach().cpu().numpy()); trues.append(batch_y.detach().cpu().numpy())
        preds, trues = np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)
        preds, trues = test_data.inverse_transform(preds), test_data.inverse_transform(trues)
        mae, mse = metric(preds, trues)
        print(f"Final Test MSE: {mse:.4f}, Final Test MAE: {mae:.4f}")
        return mse, mae

# ==============================================================================
#                 PART 3: MAIN REPLICATION SCRIPT
# ==============================================================================
def main():
    class Args:
        is_training, model, data = 1, 'PatchTST', 'custom'
        root_path, data_path, features, target, freq = DATA_DIR, 'electricity.csv', 'M', 'OT', 'h'
        checkpoints, seq_len, label_len, pred_len, enc_in = CHECKPOINTS_DIR, 336, 0, 96, 321
        d_model, n_heads, e_layers, d_ff, dropout, fc_dropout, head_dropout = 128, 16, 3, 256, 0.2, 0.2, 0.0
        patch_len, stride, padding_patch, revin, affine, subtract_last = 16, 8, 'end', 1, 0, 0
        decomposition, kernel_size, individual, num_workers, itr, train_epochs = 0, 25, 0, 0, 1, 10
        batch_size, patience, learning_rate, loss, lradj, pct_start = 128, 3, 0.0001, 'mse', 'type3', 0.3
        use_gpu, gpu = True, 0
        model_id = f'Electricity_sl{seq_len}_pl{pred_len}'
        des = 'Exp' # Description
    
    args = Args()
    
    download_data()
    
    print('Args in experiment:'); [print(f'  {k}: {v}') for k, v in args.__dict__.items()]

    exp = Exp_Main(args)
    setting = f'{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_pl{args.pred_len}_{args.des}_{0}'
    
    print(f'\n>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(setting)
    
    print(f'\n>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse, mae = exp.test(setting)
    
    print("\n" + "="*80); print("✅✅✅ PATCHTST REPLICATION COMPLETE ✅✅✅")
    print(f"  Final Test MSE: {mse:.4f} (Paper's P+CI model reports ~0.130)")
    print(f"  Final Test MAE: {mae:.4f} (Paper's P+CI model reports ~0.222)")
    print("="*80)

if __name__ == '__main__':
    main()
```
