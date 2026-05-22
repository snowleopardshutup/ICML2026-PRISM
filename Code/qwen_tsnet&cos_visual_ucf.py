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
JSON_PATH = "./ucf_descriptions1.json"
FEAT_ROOT_DIR = "./ucf_feature"
TXT_PATH = "./ucf_Testing_Videos.txt"
FEAT_NORMAL_DIR = os.path.join(FEAT_ROOT_DIR, "Testing_Normal_Videos_Anomaly")
REG_LAMBDA = 0.01

UCF_CLASSES = [
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism"
]

sys.path.append(MODEL_DIR)
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

model = Qwen3VLEmbedder(model_name_or_path=MODEL_DIR)


def get_embs(texts, prefix="A surveillance video of "):
    with torch.no_grad():
        batch = [{"text": f"{prefix}{t}"} for t in texts]
        return F.normalize(model.process(batch).to(device), p=2, dim=1)



print("\n[1/4] Replicating PRISM Initialization...")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    json_data = json.load(f)

normal_texts = json_data.get("prompt_config", {}).get(
    "normal_immunity_texts",
    ["Normal daily life"]
)
norm_embs = get_embs(normal_texts)
mu = norm_embs.mean(0, keepdim=True)

anomaly_dict = json_data["content"]["anomalies"]
prompt_overrides = json_data.get("prompt_config", {}).get("class_overrides", {})
text_anchors_raw_dict = {}
all_anom_list = []

for json_key in sorted(anomaly_dict.keys()):
    clean_override_k = json_key.lower().replace("_", " ").strip()
    cls_texts = (
        [prompt_overrides[clean_override_k]]
        if clean_override_k in prompt_overrides
        else anomaly_dict[json_key]
    )

    cls_embs = get_embs(
        cls_texts,
        prefix="A surveillance video of anomalous event: "
    )
    all_anom_list.append(cls_embs)

    key_clean = json_key.lower().replace("_", "")
    if key_clean == "roadaccident" or key_clean == "roadaccidents":
        std_name = "RoadAccidents"
    else:
        std_name = json_key.replace("_", "").capitalize()

    text_anchors_raw_dict[std_name] = cls_embs.mean(0, keepdim=True)

all_anom_embs = torch.cat(all_anom_list, dim=0)

ref_feats = torch.cat([norm_embs, all_anom_embs], dim=0)
diff_ref = ref_feats - ref_feats.mean(0)
sigma = (diff_ref.T @ diff_ref) / (len(ref_feats) - 1)

reg_sigma = sigma + REG_LAMBDA * torch.eye(sigma.shape[0]).to(device)
L, V = torch.linalg.eigh(reg_sigma)
L_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(L, min=1e-7)))
whitening_matrix = torch.matmul(torch.matmul(V, L_inv_sqrt), V.T)


def prism_text_transform(feat):

    f_white = torch.matmul(feat - mu, whitening_matrix)
    return F.normalize(f_white, p=2, dim=1)


def prism_video_transform(feat):

    f_centered = feat - mu
    return F.normalize(f_centered, p=2, dim=1)


t_anchors_raw = []
for c in UCF_CLASSES:
    if c == "Normal":
        t_anchors_raw.append(mu)
    else:
        t_anchors_raw.append(text_anchors_raw_dict.get(c, mu))

t_anchors_raw = torch.cat(t_anchors_raw, dim=0)

t_anchors_white = prism_text_transform(t_anchors_raw)


print("[2/4] Collecting frames from UCF-Crime...")

gt_dict = {}
with open(TXT_PATH, "r") as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 2:
            name = p[0].replace(".mp4", "")
            c_raw = p[1].capitalize()
            if c_raw == "Roadaccidents" or c_raw == "Roadaccident":
                c_raw = "RoadAccidents"

            regs = [
                (int(p[i]), int(p[i + 1]))
                for i in range(2, len(p) - 1, 2)
                if int(p[i]) != -1
            ]

            gt_dict[name] = {
                "class": c_raw,
                "regions": regs
            }

v_raw_list, v_white_list, v_labels = [], [], []

for vname, info in tqdm(gt_dict.items()):
    cls_name = info["class"]
    if cls_name not in UCF_CLASSES:
        continue

    path = (
        os.path.join(FEAT_ROOT_DIR, cls_name, f"{vname}.npy")
        if cls_name != "Normal"
        else os.path.join(FEAT_NORMAL_DIR, f"{vname}.npy")
    )

    if not os.path.exists(path):
        continue

    feats_np = np.load(path)
    feats_t = F.normalize(torch.from_numpy(feats_np).float().to(device), p=2, dim=1)

    is_anom = np.zeros(len(feats_np), dtype=bool)
    if cls_name != "Normal":
        for s, e in info["regions"]:
            is_anom[max(0, s // 16):min(len(feats_np), e // 16)] = True

    mask = is_anom if cls_name != "Normal" else np.ones(len(feats_np), dtype=bool)

    if mask.any():

        sel_raw = feats_t[mask][::3]

        sel_white = prism_video_transform(sel_raw)

        v_raw_list.append(sel_raw.cpu())
        v_white_list.append(sel_white.cpu())
        v_labels.extend([UCF_CLASSES.index(cls_name)] * len(sel_raw))

v_raw = torch.cat(v_raw_list, dim=0).numpy()
v_white = torch.cat(v_white_list, dim=0).numpy()
v_labels = np.array(v_labels)


print("[3/4] Generating 4K t-SNE Plot...")

fig_tsne, (ax1, ax2) = plt.subplots(1, 2, figsize=(34, 15))
cmap = plt.colormaps.get_cmap("tab20")


def plot_final_tsne(ax, v_data, t_data, title):
    combined = np.concatenate([v_data, t_data.cpu().numpy()], axis=0)

    tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=25,
        random_state=42,
        init="pca"
    )
    proj = tsne.fit_transform(combined)

    v_proj = proj[:len(v_labels)]
    t_proj = proj[len(v_labels):]

    for i, name in enumerate(UCF_CLASSES):
        m = v_labels == i
        if m.any():
            ax.scatter(
                v_proj[m, 0],
                v_proj[m, 1],
                color=cmap(i),
                alpha=0.35,
                s=30,
                label=name
            )

    for i, name in enumerate(UCF_CLASSES):
        ax.scatter(
            t_proj[i, 0],
            t_proj[i, 1],
            marker="*",
            s=1200,
            color="yellow",
            edgecolors="black",
            linewidths=3,
            zorder=10
        )
        ax.text(
            t_proj[i, 0],
            t_proj[i, 1] + 1.8,
            f" {name}",
            fontsize=15,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
            zorder=11
        )

    ax.set_title(title, fontsize=32, fontweight="bold", pad=20)
    ax.axis("off")


plot_final_tsne(
    ax1,
    v_raw,
    t_anchors_raw,
    "UCF-Crime: Before Whitening"
)

plot_final_tsne(
    ax2,
    v_white,
    t_anchors_white,
    "UCF-Crime: After PRISM (Text-Axis Aligned)"
)

leg = plt.legend(
    loc="lower center",
    bbox_to_anchor=(-0.1, -0.12),
    ncol=7,
    fontsize=20,
    markerscale=3
)

for lh in leg.legendHandles:
    lh.set_alpha(1)

plt.savefig("ucf_prism_tsne_4K.png", bbox_inches="tight", dpi=500)

print("[4/4] Generating 4K Heatmap Matrix...")


def get_full_sim_mat(v_data, t_data):
    mat = np.zeros((len(UCF_CLASSES), len(UCF_CLASSES)))
    v_t = torch.from_numpy(v_data).to(device)
    t_t = t_data.to(device)

    for i in range(len(UCF_CLASSES)):
        sub = v_t[v_labels == i]
        if len(sub) > 0:
            mat[i] = torch.matmul(sub, t_t.T).mean(0).cpu().numpy()

    return mat


sim_raw = get_full_sim_mat(v_raw, t_anchors_raw)
sim_white = get_full_sim_mat(v_white, t_anchors_white)

fig_hm, (h1, h2) = plt.subplots(1, 2, figsize=(30, 13))

heatmap_kws = dict(
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    xticklabels=UCF_CLASSES,
    yticklabels=UCF_CLASSES,
    annot_kws={"size": 15, "weight": "bold"}
)

sns.heatmap(sim_raw, ax=h1, **heatmap_kws)
h1.set_title("Similarity Matrix: Raw Features", fontsize=28, fontweight="bold", pad=20)
h1.tick_params(axis="both", which="major", labelsize=16)
plt.setp(h1.get_xticklabels(), rotation=45, ha="right")

sns.heatmap(sim_white, ax=h2, **heatmap_kws)
h2.set_title(
    "Alignment Matrix: PRISM Text-Axis Aligned",
    fontsize=28,
    fontweight="bold",
    pad=20
)
h2.tick_params(axis="both", which="major", labelsize=16)
plt.setp(h2.get_xticklabels(), rotation=45, ha="right")

plt.savefig("ucf_prism_heatmap_4K.png", bbox_inches="tight", dpi=500)

print("\nDone! Visualizations generated successfully.")