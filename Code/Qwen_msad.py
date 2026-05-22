import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# =========================================================================
# === 1. Configuration ===
# =========================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(current_dir, "Qwen3-VL-Embedding-2B")
JSON_PATH = os.path.join(current_dir, "msad_descriptions.json")
CSV_PATH = os.path.join(current_dir, "anomaly_annotation.csv")
TEST_LIST_FILE = os.path.join(current_dir, "MSAD_Test.txt")
FEAT_ANOMALY_DIR = os.path.join(current_dir, "MSAD_feature")
FEAT_NORMAL_DIR = os.path.join(current_dir, "./MSAD_feature/MSAD_normal_testing_blur")

BATCH_SIZE, FRAME_STEP = 16, 16
TOP_K, ALPHA = 10, 0.95
TEMP_A, TEMP_B, SIGMA = 0.01, 0.2, 16

# =========================================================================
# === 2. Model & Axis Initialization ===
# =========================================================================
sys.path.append(model_dir)
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
model = Qwen3VLEmbedder(model_name_or_path=model_dir)

def get_embeddings(text_list, prefix=""):
    if not text_list: return None
    text_list = [f"{prefix}{t}" for t in text_list]
    with torch.no_grad():
        embeddings = []
        for i in range(0, len(text_list), BATCH_SIZE):
            batch = [{"text": t} for t in text_list[i:i+BATCH_SIZE]]
            emb = model.process(batch).to(device)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0)

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

scenario_texts = []
for v in json_data['content']['scenarios'].values(): scenario_texts.extend(v)
norm_embs = F.normalize(get_embeddings(scenario_texts, "Normal video footage of "), p=2, dim=1)
norm_mean = torch.mean(norm_embs, dim=0)

anomaly_dict = json_data['content']['anomalies']
anom_embs_list, anom_means_list = [], []
for key in sorted(anomaly_dict.keys()):
    feats = F.normalize(get_embeddings(anomaly_dict[key], "Anomalous event: "), p=2, dim=1)
    anom_embs_list.append(feats)
    anom_means_list.append(torch.mean(feats, dim=0))
all_anom_embs = torch.cat(anom_embs_list, dim=0)

ref_diff = torch.cat([norm_embs, all_anom_embs], dim=0) - torch.cat([norm_embs, all_anom_embs], dim=0).mean(0)
sigma_mat = torch.matmul(ref_diff.T, ref_diff) / (len(ref_diff) - 1)

reg_sigma_mat = sigma_mat + 0.01 * torch.eye(sigma_mat.shape[0]).to(device)
L, V = torch.linalg.eigh(reg_sigma_mat)
L_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(L, min=1e-7)))
precision = torch.matmul(torch.matmul(V, L_inv_sqrt), V.T)
# --------------------------------------------------------

axes_b = [F.normalize(torch.matmul(precision, am - norm_mean), p=2, dim=0) for am in anom_means_list]
matrix_axes_b = torch.stack(axes_b).t()
axes_c = [F.normalize(am - norm_mean, p=2, dim=0) for am in anom_means_list]
matrix_axes_c = torch.stack(axes_c).t()
axis_d = F.normalize(torch.mean(all_anom_embs, dim=0) - norm_mean, p=2, dim=0)
memory_bank = torch.cat([norm_embs, all_anom_embs * ALPHA], dim=0).t()
memory_labels = torch.cat([torch.zeros(len(norm_embs)), torch.ones(len(all_anom_embs))]).to(device)

# =========================================================================
# === 3. Processing Loop ===
# =========================================================================
results = {'a': [], 'b': [], 'c': [], 'd': [], 'gt': [], 'is_anom_video': []}
with open(TEST_LIST_FILE, 'r') as f: test_names = [l.strip() for l in f if l.strip()]
csv_index = pd.read_csv(CSV_PATH).set_index('name').to_dict('index')

for vname in tqdm(test_names):
    is_anom = vname in csv_index
    fpath = os.path.join(FEAT_ANOMALY_DIR, vname.rsplit('_', 1)[0] if '_' in vname else vname, f"{vname}.npy") if is_anom else os.path.join(FEAT_NORMAL_DIR, f"{vname}.npy")
    if not os.path.exists(fpath): continue
    try:
        feats_np = np.load(fpath)
        if len(feats_np) == 0: continue
        ft = F.normalize(torch.from_numpy(feats_np).float().to(device), p=2, dim=1)
        
        sim = torch.matmul(ft, memory_bank)
        vals, inds = torch.topk(sim, k=TOP_K, dim=1)
        sc_a = torch.sum(F.softmax(vals/TEMP_A, dim=1) * torch.gather(memory_labels.expand(len(ft), -1), 1, inds), dim=1).cpu().numpy()
        
        lb = torch.matmul(ft - norm_mean, matrix_axes_b)
        sc_b = torch.sum(lb * F.softmax(lb/TEMP_B, dim=1), dim=1).cpu().numpy()
        lc = torch.matmul(ft - norm_mean, matrix_axes_c)
        sc_c = torch.sum(lc * F.softmax(lc/TEMP_B, dim=1), dim=1).cpu().numpy()
        sc_d = torch.matmul(ft - norm_mean, axis_d).cpu().numpy()
        
        t_frames = int(csv_index[vname]['total frames']) if is_anom else len(feats_np) * FRAME_STEP
        def post(s):
            s = np.repeat(gaussian_filter1d(s, SIGMA), FRAME_STEP)
            return s[:t_frames] if len(s) > t_frames else np.pad(s, (0, t_frames - len(s)), 'edge')

        for k, v in zip(['a','b','c','d'], [sc_a, sc_b, sc_c, sc_d]): results[k].extend(post(v))
        
        gt = np.zeros(t_frames, dtype=int)
        if is_anom:
            row = csv_index[vname]
            gt[int(row['starting frame of anomaly']):int(row['ending frame of anomaly'])] = 1
        
        results['gt'].extend(gt)
        results['is_anom_video'].extend([1 if is_anom else 0] * t_frames)
    except: continue

# =========================================================================
# === 4. Final Report (AUC, AUC_A, AP, AP_A, SNR) ===
# =========================================================================
gt_all = np.array(results['gt'])
mask_anom = np.array(results['is_anom_video']) == 1

print("\n" + "="*110)
print(f"{'MSAD PERFORMANCE BENCHMARK (Detailed Semantic Metrics)':^110}")
print("="*110)
print(f"{'Method':<28} | {'AUC':<8} | {'AUC_A':<8} | {'AP':<8} | {'AP_A':<8} | {'SNR':<8}")
print("-" * 110)

for name, key in [('A: Flashback (Memory)', 'a'), ('B: Whitening (Local)', 'b'), ('C: Raw (Fine-grained)', 'c'), ('D: Global (Coarse)', 'd')]:
    s = np.nan_to_num(np.array(results[key]))

    auc = roc_auc_score(gt_all, s)
    ap = average_precision_score(gt_all, s)

    s_a = s[mask_anom]
    gt_a = gt_all[mask_anom]
    auc_a = roc_auc_score(gt_a, s_a) if len(gt_a) > 0 else 0
    ap_a = average_precision_score(gt_a, s_a) if len(gt_a) > 0 else 0

    s_norm, s_anom = s[gt_all == 0], s[gt_all == 1]
    snr = (np.mean(s_anom) - np.mean(s_norm)) / (np.std(s_norm) + 1e-9) if len(s_anom) > 0 else 0

    print(f"{name:<28} | {auc*100:6.2f}% | {auc_a*100:6.2f}% | {ap*100:6.2f}% | {ap_a*100:6.2f}% | {snr:6.3f}")

print("="*110)