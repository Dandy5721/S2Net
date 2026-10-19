# main_bold-skipDHSRNN.py
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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../Rhythm-SNN/GSC
PROJECT_ROOT = os.path.dirname(THIS_DIR)                       # .../Rhythm-SNN
ECG_DIR = os.path.join(PROJECT_ROOT, "ECG")                    # .../Rhythm-SNN/ECG

sys.path.insert(0, PROJECT_ROOT)   # for `import ECG.xxx` used inside skipDHSRNN
sys.path.insert(0, ECG_DIR)        # for `import skipDHSRNN.xxx`
# ====== Imports ======
from bold_dataset import HCPYA, HCPA, UKB, ADNI, PPMI, NIFD
from tools.lib import dump_json, set_seed, count_parameters
from skipDHSRNN.spiking_bold_model import rnn_test
from skipDHSRNN.Hyperparameters import args

# ================= Utils =================
set_seed(1111)
device = f"cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def to_scalar_label(y):
    if torch.is_tensor(y):
        y = y.detach().cpu()
        if y.numel() == 1:
            return int(y.item())
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

def force_module_device_attr(module: nn.Module, dev: str):
    for m in module.modules():
        if hasattr(m, "device"):
            try:
                m.device = dev
            except Exception:
                pass

def infer_c_core_from_core(core: nn.Module) -> int:
    if hasattr(core, "dense_2") and hasattr(core.dense_2, "dense"):
        dense = core.dense_2.dense
        if isinstance(dense, nn.Linear):
            return int(dense.out_features)
    last_linear = None
    for m in core.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        return int(last_linear.out_features)
    raise RuntimeError("Cannot infer C_core from rnn_test core structure.")

def get_all_labels(dataset):
    ys = []
    for i in range(len(dataset)):
        _, y, _ = dataset[i]
        ys.append(to_scalar_label(y))
    return np.asarray(ys, dtype=int)

def label_count_str(y):
    uniq, cnt = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, cnt)}

# ================= MODEL CLASS =================
class BOLD_SkipDHSRNN_SubjectCLS(nn.Module):
    def __init__(
        self,
        T,
        in_dim,
        low_n,
        high_n,
        branch,
        proj_dim,
        num_classes,
        device="cuda",
    ):
        super().__init__()
        self.T = int(T)
        self.in_dim = int(in_dim)
        self.proj_dim = int(proj_dim)
        self.num_classes = int(num_classes)

        # 1) BOLD Projection: 116 -> proj_dim (e.g., 64)
        self.proj = nn.Linear(self.in_dim, self.proj_dim)

        # 2) SkipDHSRNN Core (Default in_size=4)
        self.core = rnn_test(T=self.T, low_n=low_n, high_n=high_n, branch=branch)

        # Resize the core's first-layer weights and mask to match proj_dim
        if hasattr(self.core, "rnn_1") and hasattr(self.core.rnn_1, "dense"):
            old_linear = self.core.rnn_1.dense

            # default input dim was 4
            old_input_dim = 4
            hidden_dim = old_linear.in_features - old_input_dim

            new_in_features = self.proj_dim + hidden_dim

            # resize weights
            new_linear = nn.Linear(new_in_features, old_linear.out_features, bias=(old_linear.bias is not None))
            nn.init.kaiming_uniform_(new_linear.weight, a=np.sqrt(5))
            if new_linear.bias is not None:
                nn.init.zeros_(new_linear.bias)

            new_linear.to(device)
            self.core.rnn_1.dense = new_linear

            # resize mask
            if hasattr(self.core.rnn_1, "mask"):
                old_mask = self.core.rnn_1.mask # [Out, 132]

                # slice out the recurrent mask (preserve DHSRNN topology)
                recurrent_mask = old_mask[:, old_input_dim:] # [Out, Hidden]

                # new input mask is fully connected for BOLD
                new_input_mask = torch.ones(old_linear.out_features, self.proj_dim, device=old_mask.device)

                new_mask = torch.cat([new_input_mask, recurrent_mask], dim=1) # [Out, proj_dim + Hidden]

                del self.core.rnn_1.mask
                self.core.rnn_1.register_buffer('mask', new_mask)

            if hasattr(self.core.rnn_1, "input_dim"):
                self.core.rnn_1.input_dim = new_in_features
        else:
            print("[Warning] rnn_1 structure not recognized; skipping resize.")

        force_module_device_attr(self.core, device)
        self.core.to(device)

        # 3) Classification Head
        self.c_core = infer_c_core_from_core(self.core)
        self.cls = nn.Conv1d(self.c_core, self.num_classes, kernel_size=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def apply_mask(self):
        if hasattr(self.core, "rnn_1") and hasattr(self.core.rnn_1, "apply_mask"):
            self.core.rnn_1.apply_mask()

    def forward(self, x, sc=None):
        B, T, N = x.shape
        x = x.to(self.proj.weight.device).float()

        x_flat = x.reshape(B * T, N)
        x_proj = self.proj(x_flat).reshape(B, T, self.proj_dim)
        
        core_out = self.core(x_proj)

        logits_t = self.cls(core_out)
        logits = logits_t.mean(dim=-1)
        return self.logsoftmax(logits)

# ================= Setup Experiment =================
base_exp_dir = f"./exp/skipDHSRNN_BOLD/{args.dataset}"
snn_ckp_dir = os.path.join(base_exp_dir, "checkpoint")
snn_rec_dir = os.path.join(base_exp_dir, "record")
os.makedirs(snn_ckp_dir, exist_ok=True)
os.makedirs(snn_rec_dir, exist_ok=True)

arch_name = "-".join([str(s) for s in args.fc])
train_record_base = (
    "SkipDHSRNN_BOLD_subjectCLS_T{0}_decay{1}_thr{2}_lens{3}_arch{4}_lr{5}_bran{6}_lown{7}_highn{8}"
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

    net = BOLD_SkipDHSRNN_SubjectCLS(
        T=T,
        in_dim=N,         
        low_n=args.low_n,
        high_n=args.high_n,
        branch=args.branch,
        proj_dim=64,      
        num_classes=num_classes,
        device=device,
    ).to(device)

    print("Model Params:", count_parameters(net))

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.75)

    best_metrics = {
        "val_acc": 0.0, 
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0, 
        "auc": 0.0,
        "epoch": -1
    }
    
    record_path = train_record_base.format(
        int(T), args.decay, args.thresh, args.lens, arch_name,
        args.lr, args.branch, args.low_n, args.high_n
    ) + f"_fold{fold_id+1}"
    chk_pnt_path = record_path + ".pt"

    train_acc_record, val_acc_record = [], []
    val_prec_record, val_rec_record, val_f1_record, val_auc_record = [], [], [], []
    loss_train_record, loss_val_record = [], []

    for epoch in range(args.epochs):
        net.train()
        net.apply_mask()
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
            net.apply_mask()

            train_loss += loss.item()
            pred = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_acc = 100.0 * correct / max(1, total)
        avg_train_loss = train_loss / max(1, len(train_loader))
        train_acc_record.append(train_acc)
        loss_train_record.append(avg_train_loss)

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
                "val_acc": acc, 
                "precision": precision,
                "recall": recall,
                "f1": f1, 
                "auc": auc, 
                "epoch": epoch
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
    precs = [m["precision"] for m in all_metrics]
    f1s = [m["f1"] for m in all_metrics]
    aucs = [m["auc"] for m in all_metrics]

    print("\n===== FINAL RESULTS =====")
    print(f"Mean Acc: {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}")
    print(f"Mean Pre: {np.mean(precs):.4f}")
    print(f"Mean F1 : {np.mean(f1s):.4f}")
    print(f"Mean AUC: {np.nanmean(aucs):.4f}")
    
    dump_json({"folds": all_metrics, "mean_acc": np.mean(accs)}, snn_rec_dir, f"SkipDHSRNN_BOLD_{args.dataset}_SUMMARY")