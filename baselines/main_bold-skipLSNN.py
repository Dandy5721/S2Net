# main_bold-skipLSNN.py
import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

# ====== GPU Setup ======
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Adjust as needed

# ====== Imports ======
from bold_dataset import HCPYA, HCPA, UKB, ADNI, PPMI, NIFD
from tools.lib import dump_json, set_seed, count_parameters

# Import Skip-LSNN Model
# Note: Filename in your snippet was 'spiking_bold_mode_LSNN', not 'model'
from skip_LSNN.spiking_bold_mode_LSNN import LSNN_mix
from skip_LSNN.Hyperparameters import args

# ================= Utils & Setup =================
set_seed(1111)
device = f"cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ensure the library module's global device variable is correct
import skip_LSNN.spiking_bold_mode_LSNN
skip_LSNN.spiking_bold_mode_LSNN.device = torch.device(device)

def to_scalar_label(y):
    if torch.is_tensor(y):
        y = y.detach().cpu()
        if y.numel() == 1: return int(y.item())
        return int(y.view(-1)[0].item())
    return int(y)

def compute_metrics(y_true, probs, average="macro"):
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    uniq = np.unique(y_true)
    if uniq.size < 2:
        auc = float("nan")
    else:
        try:
            auc = roc_auc_score(y_true, probs, multi_class="ovr", average=average)
        except ValueError:
            auc = float("nan")
    return acc, precision, recall, f1, auc

def label_count_str(y):
    uniq, cnt = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, cnt)}

def get_all_labels(dataset):
    ys = []
    for i in range(len(dataset)):
        _, y, _ = dataset[i]
        ys.append(to_scalar_label(y))
    return np.asarray(ys, dtype=int)

# ================= MODEL WRAPPER =================
class BOLD_SkipLSNN_SubjectCLS(nn.Module):
    """
    Wrapper for LSNN_mix to handle BOLD Subject Classification.
    """
    def __init__(self, in_dim, proj_dim, hidden_size, num_classes):
        super().__init__()
        
        # 1. Projection: 116 -> proj_dim (e.g. 64)
        self.proj = nn.Linear(in_dim, proj_dim)
        
        # 2. Skip-LSNN Core
        # LSNN_mix takes (Batch, Time, Input) -> (Batch, Class, Time)
        self.core = LSNN_mix(
            in_size=proj_dim, 
            hidden_size=hidden_size, 
            output_size=num_classes
        )
        
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x: [B, T, N]
        B, T, N = x.shape
        x = x.to(self.proj.weight.device).float()
        
        # Project: [B, T, N] -> [B, T, proj_dim]
        x_flat = x.reshape(B*T, N)
        x_proj = self.proj(x_flat).reshape(B, T, -1)
        
        # LSNN Forward
        # Returns: [B, num_classes, T]
        snn_out = self.core(x_proj)
        
        # Temporal Pooling -> [B, num_classes]
        logits = snn_out.mean(dim=-1)
        
        return self.logsoftmax(logits)

# ================= Setup Experiment =================
base_exp_dir = f"./exp/skipLSNN_BOLD/{args.dataset}"
snn_ckp_dir = os.path.join(base_exp_dir, "checkpoint")
snn_rec_dir = os.path.join(base_exp_dir, "record")
os.makedirs(snn_ckp_dir, exist_ok=True)
os.makedirs(snn_rec_dir, exist_ok=True)

arch_name = '-'.join([str(s) for s in args.fc])
skip_name = '-'.join([str(s) for s in args.skip_length])
train_record_base = 'SkipLSNN_BOLD_T{0}_arch{1}_skip{2}_lr{3}'.format(
    "MAX", arch_name, skip_name, args.lr
)

# ================= Load Dataset =================
print(f"Loading Dataset: {args.dataset}...")

if args.dataset == "HCPYA":
    dataset = HCPYA(
        data_dir="**",
        sc_root_dir="**",
        fallback_sc_path="**",
        T_fix=220, cache_sc=True,
    )
elif args.dataset == "HCPA":
    dataset = HCPA(
        data_dir="**",
        sc_root_dir="**",
        fallback_sc_path="**",
        T_fix=220, cache_sc=True,
    )
elif args.dataset == "UKB":
    dataset = UKB(
        data_dir="**",
        sc_root_dir="**",
        fallback_sc_path="**",
        T_fix=100, cache_sc=True,
    )
elif args.dataset == "ADNI":
    dataset = ADNI(
        data_dir="**",
        label_csv="**",
        sc_root_dir="**",
        fallback_sc_path="**",
        T_fix=200, cache_sc=True,
    )
elif args.dataset == "PPMI":
    dataset = PPMI(
        data_dir="**",
        label_csv="**",
        sc_root_dir="**",
        fallback_sc_path="**",
        T_fix=200, cache_sc=True,
    )
elif args.dataset == "NIFD":
    dataset = NIFD(
        data_dir="**",
        label_xlsx="**",
        sc_root_dir="**",
        fallback_sc_path="**",
        T_fix=200, cache_sc=True,
    )
else:
    raise ValueError("Unknown dataset")

T = dataset.T
N = dataset.N
num_classes = getattr(dataset, "num_classes", 7)
y_all = get_all_labels(dataset)
print(f"Dataset Loaded. T={T}, N={N}, Classes={num_classes}. Labels: {label_count_str(y_all)}")

# ================= 10-Fold Loop =================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1111)
fold_indices = [(tr, va) for tr, va in skf.split(np.zeros(len(y_all)), y_all)]

def run_one_fold(fold_id, train_idx, val_idx):
    print(f"\n========== Fold {fold_id + 1} ==========")
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model Params
    proj_dim = 64   # Map 116 -> 64
    hidden_dim = args.fc[0] if len(args.fc) > 0 else 128
    
    net = BOLD_SkipLSNN_SubjectCLS(
        in_dim=N,
        proj_dim=proj_dim,
        hidden_size=hidden_dim,
        num_classes=num_classes
    ).to(device)

    print("Model Params:", count_parameters(net))

    criterion = nn.NLLLoss()
    
    # === Optimizer with LSNN specific groups ===
    # Using param groups from your original ECG script
    base_params = [
        net.proj.weight, net.proj.bias,
        net.core.i2h.weight, net.core.i2h.bias, 
        net.core.h2h.weight, net.core.h2h.bias, 
        net.core.h2o.weight, net.core.h2o.bias
    ]
    
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': net.core.tau_adp_h, 'lr': args.lr * 3},
        {'params': net.core.tau_adp_o, 'lr': args.lr * 2}, 
    ], lr=args.lr)

    scheduler = StepLR(optimizer, step_size=100, gamma=0.75)

    best_metrics = {
        "val_acc": 0.0, "f1": 0.0, "auc": 0.0, "epoch": -1,
        "precision": 0.0, "recall": 0.0
    }
    
    record_path = train_record_base + f"_fold{fold_id+1}"
    chk_pnt_path = record_path + ".pt"

    train_acc_record, val_acc_record = [], []
    loss_train_record, loss_val_record = [], []
    val_prec_record, val_rec_record, val_f1_record, val_auc_record = [], [], [], []

    for epoch in range(args.epochs):
        # Train
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device).float()
            labels = [to_scalar_label(y) for y in labels] if isinstance(labels, (list, tuple)) else labels
            labels = torch.tensor(labels, device=device, dtype=torch.long).view(-1) if not torch.is_tensor(labels) else labels.to(device).long().view(-1)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_acc = 100.0 * correct / max(1, total)
        avg_train_loss = train_loss / max(1, len(train_loader))
        train_acc_record.append(train_acc)
        loss_train_record.append(avg_train_loss)

        # Val
        net.eval()
        val_loss = 0.0
        all_labels, all_probs = [], []
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs = inputs.to(device).float()
                labels = [to_scalar_label(y) for y in labels] if isinstance(labels, (list, tuple)) else labels
                labels = torch.tensor(labels, device=device, dtype=torch.long).view(-1) if not torch.is_tensor(labels) else labels.to(device).long().view(-1)

                outputs = net(inputs)
                val_loss += criterion(outputs, labels).item()
                all_labels.append(labels.cpu().numpy())
                all_probs.append(outputs.exp().cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        loss_val_record.append(avg_val_loss)
        
        y_true = np.concatenate(all_labels)
        y_probs = np.concatenate(all_probs)
        acc, precision, recall, f1, auc = compute_metrics(y_true, y_probs)
        
        val_acc_record.append(acc * 100.0)
        val_prec_record.append(precision)
        val_rec_record.append(recall)
        val_f1_record.append(f1)
        val_auc_record.append(auc)

        if (epoch+1) % 10 == 0:
            print(f"Fold {fold_id+1} Ep {epoch+1} | TrAcc: {train_acc:.1f}% | VaAcc: {acc*100:.1f}% | F1: {f1:.3f}")

        scheduler.step()

        if acc > best_metrics["val_acc"]:
            best_metrics.update({
                "val_acc": acc, "f1": f1, "auc": auc, "epoch": epoch,
                "precision": precision, "recall": recall
            })
            torch.save(net.state_dict(), os.path.join(snn_ckp_dir, chk_pnt_path))
            
            rec = {
                "lr": args.lr, "arch": args.fc,
                "train_loss": loss_train_record, "val_loss": loss_val_record,
                "train_acc": train_acc_record, "val_acc": val_acc_record,
                "val_prec_record": val_prec_record,
                "val_rec_record": val_rec_record,
                "val_f1_record": val_f1_record,
                "val_auc_record": val_auc_record,
                "best_metrics": best_metrics
            }
            dump_json(rec, snn_rec_dir, record_path)

    print(f"-> Best Val Acc: {best_metrics['val_acc']*100:.2f}%")
    return best_metrics

if __name__ == "__main__":
    all_metrics = []
    for fid, (tr, va) in enumerate(fold_indices):
        m = run_one_fold(fid, tr, va)
        all_metrics.append(m)

    accs = [m["val_acc"] for m in all_metrics]
    f1s = [m["f1"] for m in all_metrics]
    aucs = [m["auc"] for m in all_metrics]
    precs = [m["precision"] for m in all_metrics]

    print("\n===== FINAL RESULTS (Skip-LSNN) =====")
    print(f"Mean Acc: {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}")
    print(f"Mean Pre: {np.mean(precs):.4f}")
    print(f"Mean F1 : {np.mean(f1s):.4f}")
    print(f"Mean AUC: {np.nanmean(aucs):.4f}")
    
    dump_json({"folds": all_metrics, "mean_acc": np.mean(accs)}, snn_rec_dir, f"SkipLSNN_BOLD_{args.dataset}_SUMMARY")