# main.py
# ------------------------------------------------------------
# Entry point integrating data_module, model_module,
# train_module, utils_module. Performs both training stages,
# Weibull fitting, pseudo-label filtering, and retraining.
# ------------------------------------------------------------

import torch, os, numpy as np
import torch.optim as optim
from torchvision import transforms
from scipy.stats import weibull_min
from data_module import build_datasets, build_dataloaders
from model_module import FeatureExtractor, HealthStatusClassifier, DomainClassifier, GradientReversalLayer
from train_module import train_osbp, train_dann_with_pseudo
from utils_module import plot_confusion_matrix, plot_tsne

# ---------------- Parameters ----------------
source_path = r"C:\path\to\Source_dataset"
target_path = r"C:\path\to\Target_dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Grayscale(), transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
source_classes = sorted(os.listdir(source_path))
target_classes = sorted(os.listdir(target_path))
label_dict = {cls: i for i, cls in enumerate(source_classes)}
batch_size, unknown_label = 64, len(source_classes)
print("Device:", device)
print("Source:", source_classes)
print("Target:", target_classes)

# ---------------- Data ----------------
src_ds, tgt_ds = build_datasets(source_path, target_path, transform, label_dict, unknown_label)
loaders = build_dataloaders(src_ds, tgt_ds, batch_size)
src_loader = loaders["source_loader"]
src_train, src_test = loaders["source_train_loader"], loaders["source_test_loader"]
tgt_train, tgt_test = loaders["target_train_loader"], loaders["target_test_loader"]

# ---------------- Model init ----------------
num_classes = len(source_classes)
feat = FeatureExtractor().to(device)
clf = HealthStatusClassifier(num_classes).to(device)
disc = DomainClassifier().to(device)
grl = GradientReversalLayer(alpha=1.0)
optG, optC, optD = optim.Adam(feat.parameters(), 0.001), optim.Adam(clf.parameters(), 0.001), optim.Adam(disc.parameters(), 0.001)
epochs1, threshold = 20, 0.8
steps = len(src_loader) * epochs1

print("\n========= Stage 1: OSBP Training =========")
acc, preds, labels, feats = train_osbp(feat, clf, src_loader, tgt_train, tgt_test,
                                       optG, optC, epochs1, device, threshold)
class_names = source_classes + ["Unknown"]
plot_confusion_matrix(labels, preds, class_names)
plot_tsne(np.array(feats), np.array(labels), class_names)

# ---------------- Weibull fitting ----------------
print("Fitting Weibull models...")
correct_feats = {i: [] for i in range...continued from `main.py` (Stage 1 completed, Weibull fitting + Stage 2 training):

---

## 🧭 main.py (continued and complete)
```python
print("Fitting Weibull models...")
correct_feats = {i: [] for i in range(num_classes)}
feat.eval(); clf.eval()
with torch.no_grad():
    for x, y in src_train:
        x, y = x.to(device), y.to(device)
        f, o = feat(x), clf(feat(x))
        _, p = torch.max(o, 1)
        for i in range(len(y)):
            if p[i] == y[i]:
                correct_feats[y[i].item()].append(f[i].cpu().numpy())

avg_feat = {k: np.mean(v, 0) for k, v in correct_feats.items()}
dist = {k: [np.linalg.norm(f - avg_feat[k]) for f in v] for k, v in correct_feats.items()}

weibull_models = {}
for k, d in dist.items():
    t = np.sort(d)[-40:]
    c, loc, s = weibull_min.fit(t, floc=0)
    weibull_models[k] = {"shape": c, "scale": s}
print("✅ Weibull fitting done.")

tgt_feat, tgt_pseudo = [], []
feat.eval(); clf.eval()
with torch.no_grad():
    for x, _ in tgt_test:
        x = x.to(device)
        f = feat(x); o = clf(f)
        _, p = torch.max(o, 1)
        tgt_feat += list(f.cpu().numpy())
        tgt_pseudo += list(p.cpu().numpy())
tgt_feat, tgt_pseudo = np.array(tgt_feat), np.array(tgt_pseudo)

cdf_probs = []
for i in range(len(tgt_feat)):
    pl = tgt_pseudo[i]
    f = tgt_feat[i]; meanf = avg_feat[pl]
    dist_i = np.linalg.norm(f - meanf)
    shape, scale = weibull_models[pl]['shape'], weibull_models[pl]['scale']
    val = np.clip(dist_i / scale, 1e-10, 1e5)
    prob = 1 - np.exp(-np.clip(val ** shape, 0, 1e10))
    cdf_probs.append(prob)
cdf_probs = np.array(cdf_probs)

cdf_by_label = {}
for l, p in zip(tgt_pseudo, cdf_probs):
    cdf_by_label.setdefault(l, []).append(p)
for l in sorted(cdf_by_label):
    print(f"Label {l}, n={len(cdf_by_label[l])}, CDF:", np.round(cdf_by_label[l], 6))

threshold_by_label = {i: 1.00 for i in range(num_classes)}
unknown_flags = np.array([1 if cdf_probs[i] > threshold_by_label.get(tgt_pseudo[i], 1.0) else 0
                          for i in range(len(tgt_pseudo))])
known_idx = np.where(unknown_flags == 0)[0]
filtered_imgs, filtered_labels = [], []
feat.eval(); clf.eval()
with torch.no_grad():
    idx = 0
    for x, _ in tgt_test:
        for i in range(x.size(0)):
            if idx in known_idx:
                filtered_imgs.append(x[i].cpu().numpy())
                filtered_labels.append(tgt_pseudo[idx])
            idx += 1

fimgs = torch.tensor(np.array(filtered_imgs), dtype=torch.float32)
flabs = torch.tensor(np.array(filtered_labels), dtype=torch.long)
floader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(fimgs, flabs),
                                      batch_size=64, shuffle=True, drop_last=False)

print("\n========= Stage 2: Pseudo-Label Weighted Adversarial Training =========")
acc2, preds2, labels2, feats2 = train_dann_with_pseudo(
    feat, clf, disc, src_loader, tgt_train, floader,
    optG, optC, optD, 10, steps, device
)
plot_confusion_matrix(labels2, preds2, class_names)
plot_tsne(np.array(feats2), np.array(labels2), class_names)

if __name__ == "__main__":
    print("\n✅ All stages completed successfully.")
