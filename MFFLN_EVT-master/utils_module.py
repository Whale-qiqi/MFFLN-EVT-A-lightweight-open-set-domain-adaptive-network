# utils_module.py
# ------------------------------------------------------------
# Utility functions: entropy, adversarial loss, test, plotting,
# and visualization.
# ------------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE


def entropy(pred):
    eps = 1e-8
    pred = torch.clamp(pred, eps, 1.0)
    return -torch.sum(pred * torch.log(pred + eps), 1)


def adversarial_loss(pred, labels, entropy_weight):
    loss = F.cross_entropy(pred, labels, reduction='none')
    return (loss * (1 - entropy_weight)).mean()


def test(feat_extractor, classifier, loader, device):
    feat_extractor.eval(); classifier.eval()
    preds, labels, feats = [], [], []
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            f = feat_extractor(imgs)
            out = classifier(f)
            _, p = torch.max(out, 1)
            preds += list(p.cpu().numpy())
            labels += list(labs.cpu().numpy())
            feats += list(f.cpu().numpy())
            correct += (p == labs).sum().item(); total += labs.size(0)
    return correct / total, preds, labels, feats


def plot_confusion_matrix(true, pred, classes):
    cm = confusion_matrix(true, pred, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
    plt.title("Confusion Matrix"); plt.show()


def plot_tsne(features, labels, class_names):
    tsne = TSNE(n_components=2, random_state=42)
    ts = tsne.fit_transform(features)
    plt.figure(figsize=(8, 8))
    for i in range(len(class_names)):
        idx = [k for k, l in enumerate(labels) if l == i]
        plt.scatter(ts[idx, 0], ts[idx, 1], label=class_names[i], alpha=0.6)
    plt.legend(); plt.title("t-SNE Visualization"); plt.show()
