import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "./Qwen3-VL-Embedding-2B"
FEAT_DIR = "./xd_features"
JSON_PATH = "./xd_descriptions1.json"
# ---------------------

CLASS_NAMES = ["Normal", "Fighting", "Shooting", "Riot", "Abuse", "Car_Accident", "Explosion"]
CODE_MAP = {
    "B1": "Fighting",
    "B2": "Shooting",
    "B4": "Riot",
    "B5": "Abuse",
    "B6": "Car_Accident",
    "G": "Explosion"
}
REG_LAMBDA = 0.01


sys.path.append(MODEL_DIR)
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

model = Qwen3VLEmbedder(model_name_or_path=MODEL_DIR)


def get_embs(texts, prefix="Video footage of "):
    with torch.no_grad():
        batch = [{"text": f"{prefix}{t}"} for t in texts]
        return F.normalize(model.process(batch).to(device), p=2, dim=1)


print("\n[1/4] Modeling Statistical Background...")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    json_data = json.load(f)


norm_texts = json_data["prompt_config"]["scenario_normals"]["Generic"]
norm_embs = get_embs(norm_texts)
mu = norm_embs.mean(0, keepdim=True)

text_anchors_raw_list = []
for cls_key in ["fighting", "shooting", "riot", "abuse", "car_accident", "explosion"]:
    cls_texts = json_data["content"]["anomalies"].get(cls_key, [cls_key])
    text_anchors_raw_list.append(get_embs(cls_texts).mean(0, keepdim=True))

text_anchors_raw = torch.cat([mu] + text_anchors_raw_list, dim=0)
all_anom_embs = torch.cat(text_anchors_raw_list, dim=0)

ref_feats = torch.cat([norm_embs, all_anom_embs], dim=0)
diff_ref = ref_feats - ref_feats.mean(0)
sigma = (diff_ref.T @ diff_ref) / (len(ref_feats) - 1)

reg_sigma = sigma + REG_LAMBDA * torch.eye(sigma.shape[0]).to(device)
L, V = torch.linalg.eigh(reg_sigma)

L_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(L, min=1e-7)))


whitening_matrix = torch.matmul(torch.matmul(V, L_inv_sqrt), V.T)


def prism_text_transform(feat, mu_ref, white_mat):

    feat_white = torch.matmul(feat - mu_ref, white_mat)
    return F.normalize(feat_white, p=2, dim=1)


def prism_video_transform(feat, mu_ref):

    feat_centered = feat - mu_ref
    return F.normalize(feat_centered, p=2, dim=1)


text_anchors_white = prism_text_transform(text_anchors_raw, mu, whitening_matrix)


print("[2/4] Sampling Video Features (XD-Violence)...")

v_raw_list, v_white_list, v_labels = [], [], []
all_files = [f for f in os.listdir(FEAT_DIR) if f.endswith(".npy")]
np.random.shuffle(all_files)

counts = {name: 0 for name in CLASS_NAMES}

for f_name in tqdm(all_files):
    cls_name = "Normal"
    for code, name in CODE_MAP.items():
        if code in f_name:
            cls_name = name
            break

    if counts[cls_name] >= 150:
        continue

    feat_np = np.load(os.path.join(FEAT_DIR, f_name))
    if len(feat_np) < 5:
        continue

    feat_t = F.normalize(torch.from_numpy(feat_np).float().to(device), p=2, dim=1)

    idx = np.random.choice(len(feat_t), 5)
    sel_feat = feat_t[idx]

    v_raw_list.append(sel_feat.cpu())

    v_white_list.append(prism_video_transform(sel_feat, mu).cpu())

    v_labels.extend([CLASS_NAMES.index(cls_name)] * 5)
    counts[cls_name] += 5

    if all(v >= 150 for v in counts.values()):
        break

v_raw = torch.cat(v_raw_list, dim=0).numpy()
v_white = torch.cat(v_white_list, dim=0).numpy()
v_labels = np.array(v_labels)


print("[3/4] Running t-SNE (Joint Video-Text Space)...")


def run_joint_tsne(v_data, t_data):
    combined = np.concatenate([v_data, t_data], axis=0)
    tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=30,
        random_state=42,
        init="pca"
    )
    return tsne.fit_transform(combined)


res_raw = run_joint_tsne(v_raw, text_anchors_raw.cpu().numpy())
res_white = run_joint_tsne(v_white, text_anchors_white.cpu().numpy())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 12))
colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, 7))


def plot_side(ax, proj, title):
    v_proj = proj[:len(v_labels)]
    t_proj = proj[len(v_labels):]

    # 绘制视频特征点
    for i, name in enumerate(CLASS_NAMES):
        mask = v_labels == i
        ax.scatter(
            v_proj[mask, 0],
            v_proj[mask, 1],
            c=[colors[i]],
            alpha=0.3,
            s=40,
            label=name
        )

    # 绘制文本锚点
    ax.scatter(
        t_proj[:, 0],
        t_proj[:, 1],
        marker="*",
        s=1200,
        c="yellow",
        edgecolors="black",
        linewidths=3,
        zorder=10
    )

    for i, name in enumerate(CLASS_NAMES):
        ax.text(
            t_proj[i, 0],
            t_proj[i, 1] + 1.5,
            f" {name.upper()}",
            fontsize=16,
            fontweight="bold",
            zorder=11,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1)
        )

    ax.set_title(title, fontsize=28, fontweight="bold", pad=20)
    ax.axis("off")


plot_side(ax1, res_raw, "Before Whitening (Entangled)")
plot_side(ax2, res_white, "After PRISM (Text-Axis Aligned)")

leg = plt.legend(
    loc="lower center",
    bbox_to_anchor=(-0.1, -0.1),
    ncol=7,
    fontsize=18,
    markerscale=2
)

for lh in leg.legendHandles:
    lh.set_alpha(1)

plt.savefig("xd_prism_tsne_4K.png", bbox_inches="tight", dpi=500)


print("[4/4] Generating Alignment Heatmaps...")


def get_sim_matrix(v_data, t_data):
    v_t = torch.from_numpy(v_data).to(device)
    t_t = torch.from_numpy(t_data).to(device)
    mat = np.zeros((7, 7))

    for i in range(7):
        for j in range(7):
            v_sub = v_t[v_labels == i]
            sim = torch.matmul(v_sub, t_t[j:j + 1].T).mean().item()
            mat[i, j] = sim

    return mat


sim_raw = get_sim_matrix(v_raw, text_anchors_raw.cpu().numpy())
sim_white = get_sim_matrix(v_white, text_anchors_white.cpu().numpy())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

heatmap_kws = dict(
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    annot_kws={"size": 18, "weight": "bold"}
)

sns.heatmap(sim_raw, ax=ax1, **heatmap_kws)
ax1.set_title("Similarity Matrix: Raw", fontsize=24, fontweight="bold", pad=15)
ax1.tick_params(axis="both", which="major", labelsize=16)
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

sns.heatmap(sim_white, ax=ax2, **heatmap_kws)
ax2.set_title(
    "Alignment Matrix: PRISM Text-Axis Aligned",
    fontsize=24,
    fontweight="bold",
    pad=15
)
ax2.tick_params(axis="both", which="major", labelsize=16)
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

plt.savefig("xd_prism_heatmap_4K.png", bbox_inches="tight", dpi=500)

print("\nDone! High-resolution images saved as 'xd_prism_tsne_4K.png' and 'xd_prism_heatmap_4K.png'.")