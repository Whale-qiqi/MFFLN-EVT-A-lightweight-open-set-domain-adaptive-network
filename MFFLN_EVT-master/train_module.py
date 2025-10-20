# train_module.py
# ------------------------------------------------------------
# Training and evaluation functions: OSBP, DANN, test.
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils_module import test


def train_osbp(feature_extractor, classifier, source_train_loader,
               target_train_loader, source_test_loader,
               optimizer_G, optimizer_C, num_epochs, device, threshold):
    criterion = nn.CrossEntropyLoss()
    final_acc = final_preds = final_labels = final_feats = None
    for epoch in range(num_epochs):
        feature_extractor.train(); classifier.train()
        for (xs, ys), (xt, _) in zip(source_train_loader, target_train_loader):
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            fs, ft = feature_extractor(xs), feature_extractor(xt)
            ps, pt = classifier(fs), classifier(ft)

            loss_s = criterion(ps, ys)
            ent_loss = -torch.mean(torch.sum(pt * torch.log(pt + 1e-6), dim=1))
            max_p, pseudo = torch.max(pt[:, :-1], 1)
            unk = pt[:, -1]
            sep_loss = torch.mean((1 - unk) * (max_p > threshold).float())
            loss_tot = loss_s + 0.01 * ent_loss + 0.01 * sep_loss

            optimizer_G.zero_grad(); optimizer_C.zero_grad()
            loss_tot.backward(); optimizer_G.step(); optimizer_C.step()

        acc, preds, labels, feats = test(feature_extractor, classifier, source_test_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] | entropy={ent_loss.item():.4f} | "
              f"sep={sep_loss.item():.4f} | total={loss_tot.item():.4f} | Acc={acc*100:.2f}%")
        if epoch == num_epochs - 1:
            final_acc, final_preds, final_labels, final_feats = acc, preds, labels, feats
    return final_acc, final_preds, final_labels, final_feats


def train_dann_with_pseudo(feature_extractor, classifier, domain_disc,
                           source_loader, target_loader, test_loader,
                           optimizer_G, optimizer_C, optimizer_D,
                           num_epochs, total_steps, device):
    ce_none, ce_mean = nn.CrossEntropyLoss(reduction='none'), nn.CrossEntropyLoss(reduction='mean')
    K = None; step = 0
    for epoch in range(num_epochs):
        feature_extractor.train(); classifier.train(); domain_disc.train()
        for (xs, ys), (xt, _) in zip(source_loader, target_loader):
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            step += 1; p = step / total_steps; alpha = 2. / (1 + np.exp(-10 * p)) - 1.
            fs, ft = feature_extractor(xs), feature_extractor(xt)
            ls, lt = classifier(fs), classifier(ft)
            if K is None: K = ls.size(1)
            pt = torch.softmax(lt, 1)
            max_p, pseudo = pt.max(1)
            keep = torch.zeros_like(pseudo, dtype=torch.bool)
            for c in range(K):
                m = (pseudo == c)
                if m.any():
                    phi = max_p[m].mean(); keep |= (m & (max_p >= phi))
            ft_sel, pseudo_sel = ft[keep], pseudo[keep]
            n_s, n_tsel = xs.size(0), int(keep.sum().item())
            loss_cls = (ce_none(ls, ys).sum() + (
                ce_none(classifier(ft_sel), pseudo_sel).sum() if n_tsel > 0 else 0)
                        ) / (n_s + max(n_tsel, 1))
            ds_lab = torch.zeros(fs.size(0), dtype=torch.long, device=device)
            dt_lab = torch.ones(ft.size(0), dtype=torch.long, device=device)
            d_pred_s, d_pred_t = domain_disc(fs, alpha), domain_disc(ft, alpha)
            loss_dom_s = ce_mean(d_pred_s, ds_lab)
            H = -torch.sum(pt * torch.log(pt + 1e-8), 1); Hn = H / np.log(K)
            w_t = (1 - Hn).clamp(0, 1).detach()
            ce_t = ce_none(d_pred_t, dt_lab)
            loss_dom_t = (w_t * ce_t).sum() / w_t.sum().clamp_min(1.)
            loss_dom = loss_dom_s + loss_dom_t
            loss_tot = loss_cls + 0.01 * loss_dom

            optimizer_G.zero_grad(); optimizer_C.zero_grad(); optimizer_D.zero_grad()
            loss_tot.backward(); optimizer_G.step(); optimizer_C.step(); optimizer_D.step()

        acc, preds, labels, feats = test(feature_extractor, classifier, test_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] | L_cls={loss_cls.item():.4f} | "
              f"L_dom={loss_dom.item():.4f} | L_tot={loss_tot.item():.4f} | Acc={acc*100:.2f}%")
    return acc, preds, labels, feats
