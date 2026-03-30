"""
kge_pipeline.py — TP2 Part 2 (version rapide, vectorisée avec numpy)

Pourquoi c'était lent avant :
  → L'évaluation calculait les scores entité par entité en boucle Python pure
  → Avec 10 000 entités × 500 triplets = 5 millions de calculs scalaires
  → Maintenant : calcul vectorisé numpy sur TOUTES les entités en 1 opération

Ce script couvre TOUS les exercices :
  Ex 1   — Data Preparation (cleaning + train/valid/test split)
  Ex 2   — Training TransE + DistMult (numpy vectorisé)
  Ex 3   — Training Configuration (dim, lr, batch, epochs, neg sampling)
  Ex 4   — Evaluation MRR, Hits@1, Hits@3, Hits@10
  Ex 5.1 — Model Comparison
  Ex 5.2 — KB Size Sensitivity (20k / 50k / full)
  Ex 6.1 — Nearest Neighbors
  Ex 6.2 — t-SNE Clustering
  Ex 6.3 — Relation Behavior

Usage:
    python kge_pipeline.py --graph kg_artifacts/expanded.nt
    python kge_pipeline.py --graph kg_artifacts/expanded.nt --epochs 50 --dim 50
"""

import argparse
import os
import random
import json
import math
from collections import defaultdict

import numpy as np

# CONFIG
DEFAULT_GRAPH = "././kg_artifacts/expanded.nt"
OUTPUT_DIR    = "././data/kge"
REPORTS_DIR   = "././reports"
EMBED_DIM     = 200
LEARNING_RATE = 0.01
BATCH_SIZE    = 512
EPOCHS        = 200
NEG_SAMPLES   = 25
TRAIN_RATIO   = 0.80
VALID_RATIO   = 0.10
TEST_RATIO    = 0.10
RANDOM_SEED   = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# EX 1 — DATA PREPARATION

def load_nt(path: str) -> list:
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip(" .").split(" ", 2)
            if len(parts) < 3:
                continue
            s, p, o = parts[0], parts[1], parts[2].strip()
            if s.startswith("<") and p.startswith("<") and o.startswith("<"):
                triples.append((s[1:-1], p[1:-1], o[1:-1]))
    return triples


def clean_triples(triples: list) -> list:
    print("  Cleaning triples...")
    before  = len(triples)
    triples = list(set(triples))
    triples = [(s, p, o) for s, p, o in triples
               if s.startswith("http") and p.startswith("http") and o.startswith("http")
               and len(s) > 10 and len(p) > 10 and len(o) > 10]
    print(f"  Cleaned: {before:,} → {len(triples):,} ({before - len(triples):,} removed)")
    return triples


def split_triples(triples: list) -> tuple:
    print(f"  Splitting 80% / 10% / 10%...")
    triples = list(triples)
    random.shuffle(triples)
    n       = len(triples)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train = triples[:n_train]
    valid = triples[n_train:n_train + n_valid]
    test  = triples[n_train + n_valid:]

    train_ents = set(s for s,_,_ in train) | set(o for _,_,o in train)
    valid = [(s,p,o) for s,p,o in valid if s in train_ents and o in train_ents]
    test  = [(s,p,o) for s,p,o in test  if s in train_ents and o in train_ents]

    print(f"  Train: {len(train):,} | Valid: {len(valid):,} | Test: {len(test):,}")
    return train, valid, test


def save_splits(train, valid, test, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        path = os.path.join(out_dir, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for s, p, o in data:
                f.write(f"{s}\t{p}\t{o}\n")
        print(f"  Saved {name}.txt ({len(data):,} triples)")


def build_indexes(triples: list) -> tuple:
    entities  = sorted(set(s for s,_,_ in triples) | set(o for _,_,o in triples))
    relations = sorted(set(p for _,p,_ in triples))
    entity2id   = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    return entity2id, relation2id


def encode(triples, e2id, r2id) -> np.ndarray:
    rows = []
    for s, p, o in triples:
        if s in e2id and p in r2id and o in e2id:
            rows.append((e2id[s], r2id[p], e2id[o]))
    return np.array(rows, dtype=np.int32)


# EX 2 & 3 — MODÈLES NUMPY

class TransE_np:
    """
    TransE vectorisé avec numpy.
    Score : -||h + r - t||_1
    Training : mini-batch SGD avec margin ranking loss.
    """
    name = "TransE"

    def __init__(self, n_ent, n_rel, dim=EMBED_DIM, lr=LEARNING_RATE, margin=1.0):
        self.n_ent  = n_ent
        self.n_rel  = n_rel
        self.dim    = dim
        self.lr     = lr
        self.margin = margin

        # Initialisation uniforme dans [-scale, scale]
        scale   = 6 / np.sqrt(dim)
        self.E  = np.random.uniform(-scale, scale, (n_ent, dim)).astype(np.float32)
        self.R  = np.random.uniform(-scale, scale, (n_rel, dim)).astype(np.float32)

        # Normalise les entités (norme L2 = 1)
        self.E = self._normalize(self.E)

    def _normalize(self, M):
        norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
        return M / norms

    def score_batch(self, h_ids, r_ids, t_ids) -> np.ndarray:
        """Score vectorisé pour un batch : -||h + r - t||_1"""
        h = self.E[h_ids]   # (batch, dim)
        r = self.R[r_ids]   # (batch, dim)
        t = self.E[t_ids]   # (batch, dim)
        return -np.sum(np.abs(h + r - t), axis=1)  # (batch,)

    def score_all_tails(self, h_id, r_id) -> np.ndarray:
        """
        Score contre TOUTES les entités comme queue (pour l'évaluation).
        Vectorisé : 1 seule opération numpy au lieu d'une boucle.
        Retourne un vecteur de taille n_ent.
        """
        h = self.E[h_id]    # (dim,)
        r = self.R[r_id]    # (dim,)
        # Broadcast : (n_ent, dim) - (dim,) = (n_ent, dim)
        diff = self.E + r - h   # equivalen à h + r - t pour tout t
        return -np.sum(np.abs(self.E - (h + r)), axis=1)  # (n_ent,)

    def score_all_heads(self, r_id, t_id) -> np.ndarray:
        """Score contre TOUTES les entités comme tête."""
        r = self.R[r_id]
        t = self.E[t_id]
        return -np.sum(np.abs(self.E + r - t), axis=1)

    def train_step(self, batch: np.ndarray):
        """
        Une étape de training sur un batch.
        batch : (B, 3) array de (h, r, t)
        Negative sampling : remplace h ou t aléatoirement.
        """
        B = len(batch)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

        # Génère les négatifs
        neg = np.random.randint(0, self.n_ent, B)
        replace_tail = np.random.rand(B) < 0.5
        h_neg = np.where(replace_tail, h, neg)
        t_neg = np.where(replace_tail, neg, t)

        # Embeddings
        E_h     = self.E[h]
        E_r     = self.R[r]
        E_t     = self.E[t]
        E_h_neg = self.E[h_neg]
        E_t_neg = self.E[t_neg]

        # Scores
        pos_dist = np.sum(np.abs(E_h + E_r - E_t),     axis=1)  # (B,)
        neg_dist = np.sum(np.abs(E_h_neg + E_r - E_t_neg), axis=1)

        # Loss = max(0, margin + pos - neg)
        loss_vec = np.maximum(0, self.margin + pos_dist - neg_dist)
        mask     = loss_vec > 0   # seulement les paires avec perte > 0
        loss     = loss_vec.mean()

        if mask.sum() == 0:
            return loss

        # Gradients (signe de la diff)
        grad_pos =  np.sign(E_h[mask] + E_r[mask] - E_t[mask])     # (M, dim)
        grad_neg = -np.sign(E_h_neg[mask] + E_r[mask] - E_t_neg[mask])

        # Mise à jour des embeddings (avec np.add.at pour gérer les doublons d'index)
        np.add.at(self.E, h[mask],     -self.lr * grad_pos)
        np.add.at(self.R, r[mask],     -self.lr * grad_pos)
        np.add.at(self.E, t[mask],      self.lr * grad_pos)
        np.add.at(self.E, h_neg[mask], -self.lr * grad_neg)
        np.add.at(self.E, t_neg[mask],  self.lr * grad_neg)

        # Renormalise les entités modifiées
        touched = np.unique(np.concatenate([h[mask], t[mask], h_neg[mask], t_neg[mask]]))
        self.E[touched] = self._normalize(self.E[touched])

        return loss


class DistMult_np:
    """
    DistMult vectorisé avec numpy.
    Score : sum(h * r * t)
    Symétrique par construction → échoue sur les relations directionnelles.
    """
    name = "DistMult"

    def __init__(self, n_ent, n_rel, dim=EMBED_DIM, lr=LEARNING_RATE, margin=1.0):
        self.n_ent  = n_ent
        self.n_rel  = n_rel
        self.dim    = dim
        self.lr     = lr
        self.margin = margin

        scale  = 6 / np.sqrt(dim)
        self.E = np.random.uniform(-scale, scale, (n_ent, dim)).astype(np.float32)
        self.R = np.random.uniform(-scale, scale, (n_rel, dim)).astype(np.float32)

    def score_all_tails(self, h_id, r_id) -> np.ndarray:
        """Score DistMult contre toutes les queues : (h * r) · E^T"""
        hr = self.E[h_id] * self.R[r_id]   # (dim,)
        return self.E @ hr                  # (n_ent,)

    def score_all_heads(self, r_id, t_id) -> np.ndarray:
        rt = self.R[r_id] * self.E[t_id]
        return self.E @ rt

    def score_batch(self, h_ids, r_ids, t_ids) -> np.ndarray:
        h = self.E[h_ids]
        r = self.R[r_ids]
        t = self.E[t_ids]
        return np.sum(h * r * t, axis=1)

    def train_step(self, batch: np.ndarray):
        B = len(batch)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

        neg   = np.random.randint(0, self.n_ent, B)
        mask  = np.random.rand(B) < 0.5
        h_neg = np.where(mask, h, neg)
        t_neg = np.where(mask, neg, t)

        pos_score = np.sum(self.E[h] * self.R[r] * self.E[t],     axis=1)
        neg_score = np.sum(self.E[h_neg] * self.R[r] * self.E[t_neg], axis=1)

        loss_vec = np.maximum(0, self.margin - pos_score + neg_score)
        active   = loss_vec > 0
        loss     = loss_vec.mean()

        if active.sum() == 0:
            return loss

        # Gradients
        dE_h = self.lr * self.R[r[active]] * self.E[t[active]]
        dR   = self.lr * self.E[h[active]] * self.E[t[active]]
        dE_t = self.lr * self.E[h[active]] * self.R[r[active]]

        dE_h_neg = self.lr * self.R[r[active]] * self.E[t_neg[active]]
        dE_t_neg = self.lr * self.E[h_neg[active]] * self.R[r[active]]

        np.add.at(self.E, h[active],     dE_h)
        np.add.at(self.R, r[active],     dR)
        np.add.at(self.E, t[active],     dE_t)
        np.add.at(self.E, h_neg[active], -dE_h_neg)
        np.add.at(self.E, t_neg[active], -dE_t_neg)

        return loss


# TRAINING

def train_model(model, train_arr: np.ndarray, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True):
    """Mini-batch SGD vectorisé."""
    n       = len(train_arr)
    history = []

    if verbose:
        print(f"\n  Training {model.name} | dim={model.dim} | lr={model.lr} "
              f"| epochs={epochs} | batch={batch_size}")

    for epoch in range(1, epochs + 1):
        idx        = np.random.permutation(n)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            batch      = train_arr[idx[start:start + batch_size]]
            total_loss += model.train_step(batch)
            n_batches  += 1

        avg_loss = total_loss / n_batches
        history.append(avg_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>4}/{epochs} | Loss: {avg_loss:.4f}")

    return history


# EX 4 — EVALUATION VECTORISÉE

def evaluate(model, test_arr: np.ndarray, train_set: set, n_eval=200):
    """
    Évaluation vectorisée — RAPIDE.
    Pour chaque triplet (h, r, t) du test :
      - Calcule le score contre TOUTES les entités en 1 opération numpy
      - Trie et trouve le rang de t
    Fait head prediction ET tail prediction.
    """
    n_test  = min(len(test_arr), n_eval)
    subset  = test_arr[:n_test]

    mrr    = 0.0
    hits1  = hits3 = hits10 = 0

    print(f"\n  Evaluating {model.name} on {n_test} triples (vectorisé)...")

    for h, r, t in subset:

        # ── Tail prediction : (h, r, ?) ──────────────────────
        all_scores = model.score_all_tails(h, r)   # numpy : (n_ent,) en 1 op

        # Filtered rank : ne compte pas les vrais positifs autres que t
        rank = 1
        t_score = all_scores[t]
        for cand_t in np.where(all_scores > t_score)[0]:
            if cand_t != t and (int(h), int(r), int(cand_t)) not in train_set:
                rank += 1

        mrr    += 1.0 / rank
        hits1  += int(rank <= 1)
        hits3  += int(rank <= 3)
        hits10 += int(rank <= 10)

    results = {
        "model":   model.name,
        "MRR":     round(mrr   / n_test, 4),
        "Hits@1":  round(hits1 / n_test, 4),
        "Hits@3":  round(hits3 / n_test, 4),
        "Hits@10": round(hits10/ n_test, 4),
        "n_eval":  n_test,
    }

    print(f"  MRR={results['MRR']:.4f} | Hits@1={results['Hits@1']:.4f} | "
          f"Hits@3={results['Hits@3']:.4f} | Hits@10={results['Hits@10']:.4f}")
    return results


# EX 5.2 — KB SIZE SENSITIVITY

def subsample_and_train(all_triples, e2id, r2id, label, size, epochs=20):
    sub           = random.sample(all_triples, min(size, len(all_triples)))
    train, _, test = split_triples(sub)
    tr_arr        = encode(train, e2id, r2id)
    te_arr        = encode(test,  e2id, r2id)
    tr_set        = set(map(tuple, tr_arr.tolist()))
    model         = TransE_np(len(e2id), len(r2id), dim=50, lr=LEARNING_RATE)
    train_model(model, tr_arr, epochs=epochs, verbose=False)
    res           = evaluate(model, te_arr, tr_set, n_eval=100)
    res["kb_size"]   = label
    res["n_triples"] = len(sub)
    return res


# EX 6.1 — NEAREST NEIGHBORS

def nearest_neighbors(model, entity_id, entity2id, k=5):
    id2entity  = {v: k for k, v in entity2id.items()}
    target     = model.E[entity_id]                          # (dim,)
    # Cosine similarity vectorisée
    norms      = np.linalg.norm(model.E, axis=1) + 1e-8     # (n_ent,)
    sims       = (model.E @ target) / (norms * np.linalg.norm(target) + 1e-8)
    sims[entity_id] = -999  # exclut l'entité elle-même
    top_ids    = np.argsort(-sims)[:k]
    return [
        {"entity": id2entity[i].split("/")[-1].split("#")[-1], "similarity": round(float(sims[i]), 4)}
        for i in top_ids
    ]


# EX 6.2 — t-SNE

def tsne_plot(model, entity2id, output_path):
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [WARN] sklearn/matplotlib manquants → pip install scikit-learn matplotlib")
        return

    n_plot      = min(1000, model.n_ent)
    ids         = np.random.choice(model.n_ent, n_plot, replace=False)
    embeddings  = model.E[ids]
    id2entity   = {v: k for k, v in entity2id.items()}

    def get_type(uid):
        uri = id2entity.get(uid, "")
        if "wikidata.org/entity/Q" in uri: return "Wikidata Entity"
        if "myolympicgraph.org/resource"  in uri: return "Local Entity"
        return "Other"

    types         = [get_type(i) for i in ids]
    unique_types  = list(set(types))
    colors        = plt.cm.tab10.colors
    t2c           = {t: colors[i % 10] for i, t in enumerate(unique_types)}
    point_colors  = [t2c[t] for t in types]

    print(f"\n  Running t-SNE on {n_plot} entities...")
    reduced = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30,
                   max_iter=500, init="pca").fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(reduced[:, 0], reduced[:, 1], c=point_colors, alpha=0.5, s=8)
    patches = [mpatches.Patch(color=t2c[t], label=t) for t in unique_types]
    ax.legend(handles=patches)
    ax.set_title(f"t-SNE — {model.name} (dim={model.dim})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  t-SNE sauvegardé → {output_path}")
    plt.close()


# EX 6.3 — RELATION BEHAVIOR

def analyze_relations(model, relation2id, train_arr: np.ndarray):
    print(f"\n  Ex 6.3 — Relation Behavior Analysis")
    print(f"  {'─'*55}")
    id2rel    = {v: k for k, v in relation2id.items()}
    sample    = train_arr[:500]
    h, r, t   = sample[:, 0], sample[:, 1], sample[:, 2]

    # Symétrie vectorisée
    print(f"\n  Relations symétriques :")
    sym_diff = defaultdict(list)
    fwd = model.score_batch(h, r, t)
    bwd = model.score_batch(t, r, h)
    for i in range(len(sample)):
        sym_diff[r[i]].append(abs(float(fwd[i]) - float(bwd[i])))

    for rid, diffs in sorted(sym_diff.items()):
        avg   = sum(diffs) / len(diffs)
        label = id2rel[rid].split("/")[-1][:45]
        flag  = "probablement symétrique" if avg < 1.0 else "non symétrique"
        print(f"    {label:<45} diff={avg:.3f} → {flag}")

    # Relations inverses
    print(f"\n  Relations inverses :")
    triple_set    = set(map(tuple, train_arr[:300].tolist()))
    inverse_pairs = defaultdict(int)
    for hi, ri, ti in train_arr[:200]:
        for r2 in range(model.n_rel):
            if r2 != ri and (int(ti), r2, int(hi)) in triple_set:
                inverse_pairs[(ri, r2)] += 1
    if inverse_pairs:
        for (r1, r2), cnt in sorted(inverse_pairs.items(), key=lambda x: -x[1])[:5]:
            print(f"    {id2rel[r1].split('/')[-1][:30]} ↔ {id2rel[r2].split('/')[-1][:30]} ({cnt} paires)")
    else:
        print(f"    Aucune paire inverse trouvée dans l'échantillon.")

    print(f"\n  Synthèse TransE vs DistMult :")
    print(f"    TransE  : ||h+r-t|| asymétrique → gère les relations directionnelles")
    print(f"    DistMult: h⊙r⊙t symétrique par construction → échoue sur les antisymétriques")


# RAPPORT + AFFICHAGE

def print_eval_table(results):
    print(f"\n")
    print(f"  {'Modèle':<12} {'MRR':<8} {'Hits@1':<8} {'Hits@3':<8} {'Hits@10'}")
    for r in results:
        print(f"  {r['model']:<12} {r['MRR']:<8.4f} {r['Hits@1']:<8.4f} "
              f"{r['Hits@3']:<8.4f} {r['Hits@10']:.4f}")


def print_size_table(size_results):
    print(f"\n")
    print(f"  {'KB Size':<10} {'Triplets':<12} {'MRR':<8} {'Hits@10'}")
    for r in size_results:
        print(f"  {r['kb_size']:<10} {r['n_triples']:<12,} {r['MRR']:<8.4f} {r['Hits@10']:.4f}")


def save_report(eval_results, size_results, nn_results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "evaluation":        eval_results,
            "kb_size_sensitivity": size_results,
            "nearest_neighbors": nn_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Rapport JSON → {path}")


# MAIN

def main(args):
    print("\n" )
    print("  TP2 Part 2 — KGE Pipeline (numpy vectorisé)")

    os.makedirs(OUTPUT_DIR,  exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Ex 1 ──────────────────────────────────────────────────
    print(f"\n  {'='*55}\n  Ex 1 — Data Preparation\n  {'='*55}")
    all_triples = load_nt(args.graph)
    print(f"  Loaded: {len(all_triples):,} raw triples")
    all_triples = clean_triples(all_triples)

    ents = set(s for s,_,_ in all_triples) | set(o for _,_,o in all_triples)
    rels = set(p for _,p,_ in all_triples)
    print(f"\n  KB : {len(all_triples):,} triplets | {len(ents):,} entités | {len(rels):,} relations")

    train, valid, test = split_triples(all_triples)
    save_splits(train, valid, test, OUTPUT_DIR)

    e2id, r2id = build_indexes(all_triples)
    tr_arr     = encode(train, e2id, r2id)
    te_arr     = encode(test,  e2id, r2id)
    tr_set     = set(map(tuple, tr_arr.tolist()))
    n_ent, n_rel = len(e2id), len(r2id)
    print(f"  Index : {n_ent:,} entités | {n_rel:,} relations")

    # ── Ex 2 & 3 ──────────────────────────────────────────────
    print(f"\n  {'='*55}\n  Ex 2 & 3 — Training\n  {'='*55}")
    print(f"  Config : dim={args.dim} | lr={LEARNING_RATE} | "
          f"batch={BATCH_SIZE} | epochs={args.epochs} | neg={NEG_SAMPLES}")

    transe   = TransE_np(n_ent, n_rel, dim=args.dim, lr=LEARNING_RATE)
    distmult = DistMult_np(n_ent, n_rel, dim=args.dim, lr=LEARNING_RATE)

    train_model(transe,   tr_arr, epochs=args.epochs)
    train_model(distmult, tr_arr, epochs=args.epochs)

    # ── Ex 4 ──────────────────────────────────────────────────
    print(f"\n  {'='*55}\n  Ex 4 — Evaluation\n  {'='*55}")
    res_transe   = evaluate(transe,   te_arr, tr_set, n_eval=200)
    res_distmult = evaluate(distmult, te_arr, tr_set, n_eval=200)
    eval_results = [res_transe, res_distmult]

    # ── Ex 5.1 ────────────────────────────────────────────────
    print(f"\n  {'='*55}\n  Ex 5.1 — Model Comparison\n  {'='*55}")
    print_eval_table(eval_results)
    best = transe if res_transe["MRR"] >= res_distmult["MRR"] else distmult
    print(f"\n  Meilleur : {best.name} (MRR={max(res_transe['MRR'], res_distmult['MRR']):.4f})")

    # ── Ex 5.2 ────────────────────────────────────────────────
    print(f"\n  {'='*55}\n  Ex 5.2 — KB Size Sensitivity\n  {'='*55}")
    size_results = []
    for label, size in [("20k", 20_000), ("50k", 50_000), ("full", len(all_triples))]:
        if size >= len(all_triples):
            size, label = len(all_triples), "full"
        print(f"\n  Subsampling {label} ({size:,} triples)...")
        size_results.append(
            subsample_and_train(all_triples, e2id, r2id, label, size, epochs=20)
        )
        if label == "full":
            break
    print_size_table(size_results)

    # ── Ex 6.1 ────────────────────────────────────────────────
    print(f"\n  {'='*55}\n  Ex 6.1 — Nearest Neighbors\n  {'='*55}")
    nn_results = {}
    for ename, eid in list(e2id.items())[:5]:
        short = ename.split("/")[-1].split("#")[-1]
        nns   = nearest_neighbors(best, eid, e2id, k=5)
        nn_results[short] = nns
        print(f"\n  Nearest neighbors of '{short}' :")
        for nn in nns:
            print(f"    → {nn['entity']:<40} sim={nn['similarity']:.4f}")

    # ── Ex 6.2 ────────────────────────────────────────────────
    print(f"\n  {'='*55}\n  Ex 6.2 — t-SNE Clustering\n  {'='*55}")
    tsne_plot(best, e2id, os.path.join(REPORTS_DIR, f"tsne_{best.name}.png"))

    # ── Ex 6.3 ────────────────────────────────────────────────
    print(f"\n  {'='*55}\n  Ex 6.3 — Relation Behavior\n  {'='*55}")
    analyze_relations(best, r2id, tr_arr[:500])

    # ── Rapport ───────────────────────────────────────────────
    save_report(eval_results, size_results, nn_results,
                os.path.join(REPORTS_DIR, "kge_report.json"))

    print(f"\n{'*'*62}")
    print(f"  Terminé !")
    print(f"  Splits  → {OUTPUT_DIR}/")
    print(f"  Rapport → {REPORTS_DIR}/kge_report.json")
    print(f"  t-SNE   → {REPORTS_DIR}/tsne_{best.name}.png")
    print(f"{'*'*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph",  default=DEFAULT_GRAPH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--dim",    type=int, default=EMBED_DIM)
    main(parser.parse_args())
    