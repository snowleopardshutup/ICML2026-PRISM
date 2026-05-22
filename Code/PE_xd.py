import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# =========================================================================
# === 1. Configuration ===
# =========================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.path.dirname(os.path.abspath(__file__))

FEAT_ROOT_DIR = os.path.join(current_dir, "xd_features_pe") 
ANNOTATION_PATH = os.path.join(current_dir, "annotations.txt")
JSON_PATH = os.path.join(current_dir, "xd_descriptions1.json")
MODEL_WEIGHT_PATH = os.path.join(current_dir, "PE1", "PE-Core-L14-336.pt")

BATCH_SIZE = 16
TOP_K = 10
ALPHA = 0.95          
TEMP_A = 0.01         
TEMP_B = 0.2          
SIGMA = 16           
REG_LAMBDA = 0.01    

print("Loading PE-Core-L14-336 Model and Tokenizer...")
try:
    import core.vision_encoder.pe as pe
    from transformers import CLIPTokenizer
except ImportError:

    exit(1)

model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=False)
checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location="cpu")
state_dict = checkpoint.get("state_dict", checkpoint)
if "model" in checkpoint:
    state_dict = checkpoint["model"]
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def get_base_id(name_str):
    if "_label_" in name_str:
        return name_str.split("_label_")[0]
    return name_str

def get_embeddings(text_list, prefix=""):
    if not text_list: return None
    text_list = [f"{prefix}{t}" for t in text_list]
    with torch.no_grad():
        embeddings = []
        for i in range(0, len(text_list), BATCH_SIZE):
            batch_texts = text_list[i:i+BATCH_SIZE]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            # 使用混合精度推理
            with torch.amp.autocast('cuda', dtype=torch.float16):
                emb = model.encode_text(inputs.input_ids)
            embeddings.append(emb.float())
        return torch.cat(embeddings, dim=0)

# =========================================================================
# === 2. Initialization (Global Normal + Anomaly Statistics) ===
# =========================================================================
print("Initializing Semantics (Global Mode for XD-Violence)...")
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# --- A. Normal Embeddings (Global) ---
scenario_dict = json_data['prompt_config']['scenario_normals']
norm_texts = []
for k, v in scenario_dict.items():
    norm_texts.extend(v)
norm_embs = F.normalize(get_embeddings(norm_texts, "Video footage of "), p=2, dim=1)
global_norm_mean = torch.mean(norm_embs, dim=0)

# --- B. Anomaly Embeddings (Fine-grained) ---
code_map = json_data['prompt_config']['class_overrides']
anom_dict_json = json_data['content']['anomalies']
sorted_codes = sorted(code_map.keys())

anom_embs_list, anom_means_list = [], []
for code in sorted_codes:
    cls_name = code_map[code]
    texts = anom_dict_json.get(cls_name, [f"A video of {cls_name}"])
    feats = F.normalize(get_embeddings(texts, "Real-world video footage of anomalous event: "), p=2, dim=1)
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

# --- D. Memory Banks ---
bank_m1 = torch.cat([norm_embs, all_anom_embs], dim=0).t()
bank_m2 = torch.cat([norm_embs, all_anom_embs * ALPHA], dim=0).t()
mem_labels = torch.cat([torch.zeros(len(norm_embs)), torch.ones(len(all_anom_embs))]).to(device)

# --- E. Axes Construction ---
# M3: Global Raw
axis_global_raw = F.normalize(global_anom_mean - global_norm_mean, p=2, dim=0)
# M4: Global Whitened
axis_global_white = F.normalize(torch.matmul(precision, global_anom_mean - global_norm_mean), p=2, dim=0)
# M5: Multi-Axis Raw
axes_multi_raw = torch.stack([F.normalize(am - global_norm_mean, p=2, dim=0) for am in anom_means_list]).t()
# M6: PRISM (Multi-Axis Whitened)
axes_multi_white = torch.stack([F.normalize(torch.matmul(precision, am - global_norm_mean), p=2, dim=0) for am in anom_means_list]).t()

print(f"Initialization Complete. Global Noise Floor modeled with {len(ref_feats)} text vectors.")

# =========================================================================
# === 3. Inference Loop ===
# =========================================================================
gt_dict = {}
if os.path.exists(ANNOTATION_PATH):
    with open(ANNOTATION_PATH, 'r') as f:
        for line in f:
            p = line.strip().split()
            if not p: continue
            base_id = get_base_id(p[0])
            regions = [(int(p[i]), int(p[i+1])) for i in range(1, len(p)-1, 2)]
            gt_dict[base_id] = regions

all_tasks = [f for f in os.listdir(FEAT_ROOT_DIR) if f.endswith('.npy')]
res = {k: [] for k in ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'gt', 'is_anom_video']}

for f_name in tqdm(all_tasks):
    vname_base = get_base_id(os.path.splitext(f_name)[0])
    path = os.path.join(FEAT_ROOT_DIR, f_name)
    
    try:
        feats_t = F.normalize(torch.from_numpy(np.load(path)).float().to(device), p=2, dim=1)
        feats_cen = feats_t - global_norm_mean 

        # --- M1/M2: Memory ---
        sim_m1 = torch.matmul(feats_t, bank_m1)
        v1, i1 = torch.topk(sim_m1, k=TOP_K, dim=1)
        s_m1 = torch.sum(F.softmax(v1/TEMP_A, dim=1) * torch.gather(mem_labels.expand(len(feats_t), -1), 1, i1), dim=1)
        
        sim_m2 = torch.matmul(feats_t, bank_m2)
        v2, i2 = torch.topk(sim_m2, k=TOP_K, dim=1)
        s_m2 = torch.sum(F.softmax(v2/TEMP_A, dim=1) * torch.gather(mem_labels.expand(len(feats_t), -1), 1, i2), dim=1)

        # --- M3/M4: Global Projection ---
        s_m3 = torch.matmul(feats_cen, axis_global_raw)
        s_m4 = torch.matmul(feats_cen, axis_global_white)

        # --- M5/M6: Multi-Axis (PRISM Logic) ---
        l_m5 = torch.matmul(feats_cen, axes_multi_raw)
        s_m5 = torch.sum(l_m5 * F.softmax(l_m5/TEMP_B, dim=1), dim=1)
        
        l_m6 = torch.matmul(feats_cen, axes_multi_white)
        s_m6 = torch.sum(l_m6 * F.softmax(l_m6/TEMP_B, dim=1), dim=1)

        # GT Loading
        gt_seq = np.zeros(len(feats_t), dtype=int)
        is_anom = 1 if vname_base in gt_dict else 0
        if is_anom:
            for s, e in gt_dict[vname_base]:
                gt_seq[max(0, s//16):min(len(feats_t), e//16)] = 1
        
        for k, v in zip(['m1','m2','m3','m4','m5','m6'], [s_m1, s_m2, s_m3, s_m4, s_m5, s_m6]):
            res[k].extend(gaussian_filter1d(v.cpu().numpy(), SIGMA))
        res['gt'].extend(gt_seq)
        res['is_anom_video'].extend([is_anom] * len(feats_t))
        
    except Exception as e:
        print(f"Error processing {f_name}: {e}")
        continue

# =========================================================================
# === 4. Reporting ===
# =========================================================================
labels = np.array(res['gt'])
mask_anom = np.array(res['is_anom_video']) == 1

print("\n" + "="*120)
print(f"{'XD-VIOLENCE GLOBAL STATISTICS ABLATION STUDY (PE-Core-L14-336)':^120}")
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
    auc_a = roc_auc_score(gt_a, s_a) if len(gt_a) > 0 and len(np.unique(gt_a)) > 1 else 0
    sn_n, sn_a = s[labels==0], s[labels==1]
    snr = (np.mean(sn_a)-np.mean(sn_n))/(np.std(sn_n)+1e-9) if len(sn_a)>0 else 0
    print(f"{key.upper():<4} | {name:<25} | {white:<10} | {gran:<12} | {auc*100:6.2f}% | {auc_a*100:6.2f}% | {ap*100:6.2f}% | {snr:6.3f}")
print("="*120)