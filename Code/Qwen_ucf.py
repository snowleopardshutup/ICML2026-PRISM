import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(current_dir, "Qwen3-VL-Embedding-2B")
JSON_PATH = os.path.join(current_dir, "ucf_descriptions1.json") 
TXT_PATH = os.path.join(current_dir, "ucf_Testing_Videos.txt")
FEAT_ROOT_DIR = os.path.join(current_dir, "ucf_feature")
FEAT_NORMAL_DIR = os.path.join(FEAT_ROOT_DIR, "Testing_Normal_Videos_Anomaly")


BATCH_SIZE = 16
FRAME_STEP = 16
TOP_K = 10
ALPHA = 0.95          # Flashback Scaling
TEMP_A = 0.01         # Memory Temperature
TEMP_B = 0.2          # Projection Temperature
SIGMA = 16            # Smoothing
REG_LAMBDA = 0.01     # Whitening Regularization

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

print("Initializing Semantics...")
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# --- A. Normal Embeddings ---
normal_texts = json_data.get('prompt_config', {}).get('normal_immunity_texts', ["Normal daily life"])
norm_embs = F.normalize(get_embeddings(normal_texts, "A surveillance video of "), p=2, dim=1)
norm_mean = torch.mean(norm_embs, dim=0) 

# --- B. Anomaly Embeddings ---
anomaly_dict = json_data['content']['anomalies']
prompt_overrides = json_data.get('prompt_config', {}).get('class_overrides', {})
anom_embs_list, anom_means_list = [], []

for k in sorted(anomaly_dict.keys()):
    clean_k = k.lower().replace('_', ' ').strip()
    txt = [prompt_overrides[clean_k]] if clean_k in prompt_overrides else anomaly_dict[k]
    feats = F.normalize(get_embeddings(txt, "A surveillance video of anomalous event: "), p=2, dim=1)
    anom_embs_list.append(feats)
    anom_means_list.append(torch.mean(feats, dim=0))

all_anom_embs = torch.cat(anom_embs_list, dim=0)
global_anom_mean = torch.mean(all_anom_embs, dim=0)


ref_feats = torch.cat([norm_embs, all_anom_embs], dim=0)
diff_ref = ref_feats - ref_feats.mean(0)
sigma_mat = torch.matmul(diff_ref.T, diff_ref) / (len(ref_feats) - 1)

reg_sigma_mat = sigma_mat + REG_LAMBDA * torch.eye(sigma_mat.shape[0]).to(device)
L, V = torch.linalg.eigh(reg_sigma_mat)
L_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(L, min=1e-7)))
precision = torch.matmul(torch.matmul(V, L_inv_sqrt), V.T)

# --- D. Memory Banks (Strict Match) ---
# M1: Raw Bank
bank_m1 = torch.cat([norm_embs, all_anom_embs], dim=0).t()
# M2: Penalized Bank (Anomaly * 0.95)
bank_m2 = torch.cat([norm_embs, all_anom_embs * ALPHA], dim=0).t()
# Labels
mem_labels = torch.cat([torch.zeros(len(norm_embs)), torch.ones(len(all_anom_embs))]).to(device)

# --- E. Axes Construction ---
# M3: Global Raw
axis_global_raw = F.normalize(global_anom_mean - norm_mean, p=2, dim=0)

# M4: Global Whitened (Apply precision / whitening matrix)
axis_global_white = F.normalize(torch.matmul(precision, global_anom_mean - norm_mean), p=2, dim=0)

# M5: Multi-Axis Raw (Fine-grained)
axes_multi_raw = torch.stack([F.normalize(am - norm_mean, p=2, dim=0) for am in anom_means_list]).t()

# M6: Multi-Axis Whitened (Fine-grained + Precision / Whitening) - PRISM
axes_multi_white = torch.stack([F.normalize(torch.matmul(precision, am - norm_mean), p=2, dim=0) for am in anom_means_list]).t()

print("Initialization Complete.")

# =========================================================================
# === 3. Inference Loop ===
# =========================================================================
gt_dict = {}
if os.path.exists(TXT_PATH):
    with open(TXT_PATH, 'r') as f:
        for line in f:
            p = line.strip().split()
            if not p: continue
            name = p[0].replace('.mp4', '')
            regs = [(int(p[i]), int(p[i+1])) for i in range(2, len(p)-1, 2) if int(p[i]) != -1]
            gt_dict[name] = {'class': p[1], 'regions': regs}
else:
    print(f"Error: TXT_PATH not found: {TXT_PATH}")

all_tasks = []
for name, info in gt_dict.items():
    if info['class'] != 'Normal': all_tasks.append((name, info['class'], info['regions'], FEAT_ROOT_DIR))
if os.path.exists(FEAT_NORMAL_DIR):
    for f in os.listdir(FEAT_NORMAL_DIR):
        if f.endswith('.npy'): all_tasks.append((f.replace('.npy',''), 'Normal', [], FEAT_NORMAL_DIR))

if len(all_tasks) == 0:
    raise ValueError("No video tasks found! Check dataset paths.")

res = {k: [] for k in ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'gt', 'is_anom_video']}

print(f"Starting processing on {len(all_tasks)} videos...")

for vname, cls_name, regions, root_dir in tqdm(all_tasks):
    path = os.path.join(root_dir, cls_name, f"{vname}.npy") if cls_name != 'Normal' else os.path.join(root_dir, f"{vname}.npy")
    if not os.path.exists(path): continue
    
    # Load and Normalize
    feats_t = F.normalize(torch.from_numpy(np.load(path)).float().to(device), p=2, dim=1)
    # Center features (Critical for projection methods)
    feats_cen = feats_t - norm_mean
    
    # --- M1: Memory Raw ---
    sim_m1 = torch.matmul(feats_t, bank_m1)
    v1, i1 = torch.topk(sim_m1, k=TOP_K, dim=1)
    s_m1 = torch.sum(F.softmax(v1/TEMP_A, dim=1) * torch.gather(mem_labels.expand(len(feats_t), -1), 1, i1), dim=1)
    
    # --- M2: Flashback (With Anomaly Penalty) ---
    sim_m2 = torch.matmul(feats_t, bank_m2)
    v2, i2 = torch.topk(sim_m2, k=TOP_K, dim=1)
    s_m2 = torch.sum(F.softmax(v2/TEMP_A, dim=1) * torch.gather(mem_labels.expand(len(feats_t), -1), 1, i2), dim=1)

    # --- M3: Global Raw ---
    s_m3 = torch.matmul(feats_cen, axis_global_raw)
    
    # --- M4: Global Whitened ---
    s_m4 = torch.matmul(feats_cen, axis_global_white)
    
    # --- M5: Multi Raw ---
    l_m5 = torch.matmul(feats_cen, axes_multi_raw)
    s_m5 = torch.sum(l_m5 * F.softmax(l_m5/TEMP_B, dim=1), dim=1)
    
    # --- M6: PRISM (Full Method) ---
    l_m6 = torch.matmul(feats_cen, axes_multi_white)
    s_m6 = torch.sum(l_m6 * F.softmax(l_m6/TEMP_B, dim=1), dim=1)

    # GT & Store
    gt = np.zeros(len(feats_t), dtype=int)
    for s, e in regions: gt[max(0, s//FRAME_STEP):min(len(feats_t), e//FRAME_STEP)] = 1
    
    for k, v in zip(['m1','m2','m3','m4','m5','m6'], [s_m1, s_m2, s_m3, s_m4, s_m5, s_m6]):
        res[k].extend(gaussian_filter1d(v.cpu().numpy(), SIGMA))
    res['gt'].extend(gt)
    res['is_anom_video'].extend([1 if cls_name != 'Normal' else 0] * len(feats_t))

# =========================================================================
# === 4. Report Generation ===
# =========================================================================
labels = np.array(res['gt'])
mask_anom = np.array(res['is_anom_video']) == 1

if len(labels) == 0:
    print("Error: No results collected.")
else:
    print("\n" + "="*120)
    print(f"{'UCF-CRIME 6-ROW ABLATION STUDY (REPLICATED LOGIC)':^120}")
    print("="*120)
    print(f"{'ID':<4} | {'Method Name':<25} | {'Whitening?':<10} | {'Granularity':<12} | {'AUC':<8} | {'AUC_A':<8} | {'AP':<8} | {'SNR':<8}")
    print("-" * 120)

    configs = [
        ('m1', 'Memory (Raw)', 'No', 'Instance'),
        ('m2', 'Memory (Flashback)', 'No', 'Instance'),
        ('m3', 'Global (Raw)', 'No', 'Global'),
        ('m4', 'Global (Whitened)', 'Yes', 'Global'),
        ('m5', 'Multi-Axis (Raw)', 'No', 'Fine-grained'),
        ('m6', 'PRISM (Full)', 'Yes', 'Fine-grained')
    ]

    for key, name, white, gran in configs:
        s = np.nan_to_num(np.array(res[key]))
        auc = roc_auc_score(labels, s)
        ap = average_precision_score(labels, s)
        s_a, gt_a = s[mask_anom], labels[mask_anom]
        auc_a = roc_auc_score(gt_a, s_a) if len(gt_a) > 0 else 0
        sn_n, sn_a = s[labels==0], s[labels==1]
        snr = (np.mean(sn_a)-np.mean(sn_n))/(np.std(sn_n)+1e-9) if len(sn_a)>0 else 0
        
        print(f"{key.upper():<4} | {name:<25} | {white:<10} | {gran:<12} | {auc*100:6.2f}% | {auc_a*100:6.2f}% | {ap*100:6.2f}% | {snr:6.3f}")
    print("="*120)