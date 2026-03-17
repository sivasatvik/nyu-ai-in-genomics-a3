# %% [markdown]
# # Assignment 3: Cross-Species Foundation Models for Transcription Factors
# 
# #### Siva Satvik Mandapati - sm12779

# %%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
import os
import json

from pathlib import Path

# Non-interactive plotting for batch runs / script export
plt.ioff()

# Force line-buffered stdout so print statements appear immediately in sbatch /
# Singularity logs rather than being flushed only at script exit.
sys.stdout.reconfigure(line_buffering=True)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# %% [markdown]
# ## 1. Build the Multi-Species TF Dataset
# ### 1.1 Retrieve TF gene sets

# %%
import mygene
import requests
import time

# GO:0003700 = DNA-binding transcription factor activity (as specified in the assignment)
GO_TERMS = ["GO:0003700"]
SPECIES_ORDER = ["human", "mouse", "fruitfly"]
SPECIES_CODES = {
    "human": "homo_sapiens",
    "mouse": "mus_musculus",
    "fruitfly": "drosophila_melanogaster",
}

MG = mygene.MyGeneInfo()

def fetch_tf_genes(species: str, max_genes: int) -> pd.DataFrame:
    """Fetch genes annotated to TF GO terms for a given species."""
    res = MG.query(
        q=" OR ".join(GO_TERMS),
        species=species,
        fields="symbol,name,entrezgene,ensembl.gene",
        size=1000,
    )
    hits = res.get("hits", [])
    rows = []
    for hit in hits:
        ensembl_entries = hit.get("ensembl")
        if isinstance(ensembl_entries, dict):
            ensembl_entries = [ensembl_entries]
        for entry in ensembl_entries or []:
            gene_id = entry.get("gene")
            if not gene_id:
                continue
            rows.append({
                "species": species,
                "symbol": hit.get("symbol"),
                "name": hit.get("name"),
                "entrez": hit.get("entrezgene"),
                "ensembl_gene_id": gene_id,
                "is_tf": 1,
            })
    df = pd.DataFrame(rows).drop_duplicates("ensembl_gene_id") if rows else pd.DataFrame(
        columns=["species", "symbol", "name", "entrez", "ensembl_gene_id", "is_tf"]
    )
    if len(df) > max_genes:
        df = df.sample(max_genes, random_state=SEED)
    print(f"[INFO] fetch_tf_genes({species}): {len(df)} TF genes found.")
    return df


# %% [markdown]
# ### 1.2 Sample non-TF genes

# %%
def sample_non_tf_genes(species: str, n_samples: int) -> pd.DataFrame:
    """Sample genes annotated to GO:0003674 (molecular function) but lacking TF GO terms."""
    res = MG.query(
        q="GO:0003674",
        species=species,
        fields="symbol,name,entrezgene,ensembl.gene,go.MF",
        size=1000,
    )
    hits = res.get("hits", [])
    rows = []
    for hit in hits:
        go_field = hit.get("go")
        mf_entries = go_field.get("MF", []) if isinstance(go_field, dict) else []
        if isinstance(mf_entries, dict):
            mf_entries = [mf_entries]
        go_terms_ids = {
            entry.get("id") for entry in mf_entries
            if isinstance(entry, dict) and entry.get("id")
        }
        # Exclude any gene that has a TF GO term
        if go_terms_ids.intersection(set(GO_TERMS)):
            continue
        ensembl_entries = hit.get("ensembl")
        if isinstance(ensembl_entries, dict):
            ensembl_entries = [ensembl_entries]
        for entry in ensembl_entries or []:
            gene_id = entry.get("gene")
            if not gene_id:
                continue
            rows.append({
                "species": species,
                "symbol": hit.get("symbol"),
                "name": hit.get("name"),
                "entrez": hit.get("entrezgene"),
                "ensembl_gene_id": gene_id,
                "is_tf": 0,
            })
    df = pd.DataFrame(rows).drop_duplicates("ensembl_gene_id") if rows else pd.DataFrame(
        columns=["species", "symbol", "name", "entrez", "ensembl_gene_id", "is_tf"]
    )
    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=SEED)
    print(f"[INFO] sample_non_tf_genes({species}): {len(df)} non-TF genes sampled.")
    return df


# %% [markdown]
# ### 1.3 Retrieve promoter, CDS, and protein sequences

# %%
def fetch_gene_metadata(ensembl_gene_id: str):
    url = f"https://rest.ensembl.org/lookup/id/{ensembl_gene_id}?expand=1"
    resp = requests.get(url, headers={"Content-Type": "application/json"}, timeout=20)
    return resp.json() if resp.ok else None


def fetch_sequence(url: str):
    resp = requests.get(url, headers={"Content-Type": "text/plain"}, timeout=20)
    return resp.text.strip() if resp.ok else None


def build_sequences(df, species, promoter_flank=2000, throttle_sec=0.2):
    rows = []
    for _, row in df.iterrows():
        meta = fetch_gene_metadata(row["ensembl_gene_id"])
        if not meta:
            continue

        chrom = meta.get("seq_region_name")
        strand = meta.get("strand", 1)
        tss = meta.get("start") if strand == 1 else meta.get("end")
        if chrom is None or tss is None:
            continue

        start = max(1, tss - promoter_flank)
        end = tss + promoter_flank
        promoter_url = (
            f"https://rest.ensembl.org/sequence/region/"
            f"{SPECIES_CODES[species]}/{chrom}:{start}..{end}:{strand}"
        )
        promoter_seq = fetch_sequence(promoter_url)
        if not promoter_seq:
            continue

        canonical = next(
            (tx for tx in meta.get("Transcript", []) if tx.get("is_canonical")), None
        )
        if not canonical:
            continue

        transcript_id = canonical.get("id")
        cds_seq = (
            fetch_sequence(f"https://rest.ensembl.org/sequence/id/{transcript_id}?type=cds")
            if transcript_id else None
        )

        protein_id = (canonical.get("Translation") or {}).get("id")
        protein_seq = (
            fetch_sequence(f"https://rest.ensembl.org/sequence/id/{protein_id}?type=protein")
            if protein_id else None
        )

        if not cds_seq or not protein_seq:
            continue

        rows.append({
            "species": species,
            "symbol": row["symbol"],
            "ensembl_gene_id": row["ensembl_gene_id"],
            "is_tf": row["is_tf"],
            "cds_seq": cds_seq,
            "protein_seq": protein_seq,
            "chrom": chrom,
            "strand": strand,
        })
        time.sleep(throttle_sec)

    return pd.DataFrame(rows)


# %% [markdown]
# ### 1.4 Build full dataset

# %%
def build_tf_dataset(per_species_limit=300, promoter_flank=2000, throttle_sec=0.2):
    records = []

    for species in SPECIES_ORDER:
        tf_df = fetch_tf_genes(species, per_species_limit)
        if tf_df.empty:
            print(f"[WARN] No TF genes fetched for {species}, skipping.")
            continue

        non_tf_df = sample_non_tf_genes(species, n_samples=len(tf_df))
        combined = pd.concat([tf_df, non_tf_df], ignore_index=True)
        seq_df = build_sequences(combined, species, promoter_flank, throttle_sec)
        print(f"[INFO] Collected {len(seq_df)} sequence records for {species}")
        records.append(seq_df)

    if not records:
        raise RuntimeError("No records collected; check network/API availability.")

    dataset = pd.concat(records, ignore_index=True)
    out_dir = DATA_DIR
    parquet_path = out_dir / "tf_multispecies_sequences_large.parquet"
    dataset.to_parquet(parquet_path, index=False)
    print(f"[INFO] Saved dataset with shape {dataset.shape} to {parquet_path}")

    splits_meta = {
        "human_all": dataset.index[dataset["species"] == "human"].tolist(),
        "mouse_all": dataset.index[dataset["species"] == "mouse"].tolist(),
        "fly_all": dataset.index[dataset["species"] == "fruitfly"].tolist(),
    }
    with open(out_dir / "tf_split_indices.json", "w") as fh:
        json.dump(splits_meta, fh, indent=2)
    print("[INFO] Wrote split index file with keys:", ", ".join(splits_meta.keys()))
    return dataset

# Load dataset if already built, otherwise build it
parquet_path = DATA_DIR / "tf_multispecies_sequences_large.parquet"
if parquet_path.exists():
    dataset = pd.read_parquet(parquet_path)
    print(f"[INFO] Loaded existing dataset with shape {dataset.shape}")
else:
    dataset = build_tf_dataset()

dataset.head()


# %% [markdown]
# ## 1.5 Dataset Statistics + Visualizations
# ### Gene counts per species and label

# %%
# Gene count table per species and label
count_table = dataset.groupby(["species", "is_tf"]).size().unstack(fill_value=0)
count_table.columns = ["non-TF (0)", "TF (1)"]
count_table["Total"] = count_table.sum(axis=1)
print("Gene counts per species and label:")
print(count_table)

# CDS and protein lengths
dataset["cds_len"] = dataset["cds_seq"].str.len()
dataset["protein_len"] = dataset["protein_seq"].str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.violinplot(data=dataset, x="species", y="cds_len", ax=axes[0])
axes[0].set_title("CDS Length Distribution per Species")
axes[0].set_ylabel("CDS length (bp)")

sns.violinplot(data=dataset, x="species", y="protein_len", ax=axes[1])
axes[1].set_title("Protein Length Distribution per Species")
axes[1].set_ylabel("Protein length (aa)")

plt.tight_layout()
fig_path = FIG_DIR / "cds_protein_length_distributions.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved figure: {fig_path}")


# %% [markdown]
# ### GC content and amino-acid composition

# %%
def gc_content(seq):
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / max(len(seq), 1)

dataset["gc"] = dataset["cds_seq"].apply(gc_content)

fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=dataset, x="species", y="gc", hue="is_tf", ax=ax)
ax.set_title("GC Content per Species (TF vs non-TF)")
plt.tight_layout()
fig_path = FIG_DIR / "gc_content_by_species_tf_vs_nontf.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved figure: {fig_path}")

# Amino-acid composition (fraction of each of 20 standard AA per protein)
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

def aa_composition(seq):
    seq = seq.upper()
    total = max(len(seq), 1)
    return {aa: seq.count(aa) / total for aa in AA_ALPHABET}

aa_df = pd.DataFrame([aa_composition(s) for s in dataset["protein_seq"]])
aa_df["species"] = dataset["species"].values
aa_df["is_tf"] = dataset["is_tf"].values

# Mean AA composition by species
aa_mean = aa_df.groupby("species")[AA_ALPHABET].mean()
print("Mean amino-acid composition per species (first 5 AAs shown):")
print(aa_mean[AA_ALPHABET[:5]].round(4))

fig, ax = plt.subplots(figsize=(14, 4))
aa_mean.T.plot(kind="bar", ax=ax)
ax.set_title("Mean Amino-Acid Composition per Species")
ax.set_xlabel("Amino acid")
ax.set_ylabel("Mean fraction")
ax.legend(title="Species")
plt.tight_layout()
fig_path = FIG_DIR / "mean_amino_acid_composition_by_species.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved figure: {fig_path}")


# %% [markdown]
# ### 1.4 Dataset Snapshot – Observations
# 
# - **CDS length**: Human and mouse CDS sequences tend to be longer than fly, reflecting the larger and more complex genomes of mammals. TF genes often have longer CDS than non-TF genes.
# - **Protein length**: Mirrors CDS length trends. Fly proteins are generally shorter on average.
# - **GC content**: Human and mouse CDS sequences have higher GC content (~50–55%) than fly (~42–48%), consistent with known genome-wide GC differences between vertebrates and insects.
# - **Amino-acid composition**: Differences in serine (S), glycine (G), and glutamic acid (E) fractions across species are notable. TF proteins tend to be enriched in charged/polar residues (K, R, E) compared to non-TFs, which is consistent with their DNA-binding role.
# 

# %% [markdown]
# ## 1.6 Train/Val/Test Splits

# %%
from sklearn.model_selection import train_test_split

splits = {}

# Per-species indices
human_idx = np.array(dataset.index[dataset["species"] == "human"].tolist())
mouse_idx = np.array(dataset.index[dataset["species"] == "mouse"].tolist())
fly_idx   = np.array(dataset.index[dataset["species"] == "fruitfly"].tolist())

y_all = dataset["is_tf"].values

# -----------------------------------------------------------------------
# Split 1: Human-focused
# Train on 70% of human, validate on 15% of human, test on mouse + fly
# -----------------------------------------------------------------------
h_train, h_val = train_test_split(human_idx, test_size=0.3, random_state=SEED,
                                   stratify=y_all[human_idx])
h_val, h_test_human = train_test_split(h_val, test_size=0.5, random_state=SEED,
                                        stratify=y_all[h_val])

splits["split1_train"] = h_train.tolist()
splits["split1_val"]   = h_val.tolist()
splits["split1_test"]  = mouse_idx.tolist() + fly_idx.tolist()

# -----------------------------------------------------------------------
# Split 2: Mouse few-shot
# Train on all human + 50 mouse genes; validate on next 50 mouse; test rest of mouse + all fly
# -----------------------------------------------------------------------
np.random.seed(SEED)
mouse_shuffled = np.random.permutation(mouse_idx)
mouse_fewshot  = mouse_shuffled[:50]
mouse_fewval   = mouse_shuffled[50:100]
mouse_test     = mouse_shuffled[100:]

splits["split2_train"] = human_idx.tolist() + mouse_fewshot.tolist()
splits["split2_val"]   = mouse_fewval.tolist()
splits["split2_test"]  = mouse_test.tolist() + fly_idx.tolist()

# -----------------------------------------------------------------------
# Split 3: Fly hold-out
# Train on 80% of (human + mouse), validate on 20% of (human + mouse), test on all fly
# -----------------------------------------------------------------------
hm_idx = np.concatenate([human_idx, mouse_idx])
hm_train, hm_val = train_test_split(hm_idx, test_size=0.2, random_state=SEED,
                                     stratify=y_all[hm_idx])

splits["split3_train"] = hm_train.tolist()
splits["split3_val"]   = hm_val.tolist()
splits["split3_test"]  = fly_idx.tolist()

# Save splits
with open(DATA_DIR / "tf_split_indices.json", "w") as f:
    json.dump(splits, f, indent=2)

print("Split sizes:")
for k, v in splits.items():
    print(f"  {k}: {len(v)} samples")


# %% [markdown]
# ## 2. DNA / Protein Foundation Model Embeddings

# %% [markdown]
# ### 2.1 Nucleotide Transformer (CDS embeddings)

# %%
from transformers import AutoTokenizer, AutoModel, AutoConfig

dna_model_repo = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
dna_model_local = Path("models/nucleotide-transformer-v2-500m-multi-species")
dna_model_name = str(dna_model_local) if dna_model_local.exists() else dna_model_repo
print(f"[INFO] Loading DNA model from {'local folder' if dna_model_local.exists() else 'Hugging Face'}: {dna_model_name}")

dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_name, trust_remote_code=True)
dna_config = AutoConfig.from_pretrained(dna_model_name, trust_remote_code=True)
if not hasattr(dna_config, "is_decoder"):
    dna_config.is_decoder = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dna_model = AutoModel.from_pretrained(
    dna_model_name,
    config=dna_config,
    trust_remote_code=True,
).eval().to(device)
if torch.cuda.is_available():
    dna_model = dna_model.half()  # fp16 on GPU

MAX_DNA_LEN = 1024
STRIDE = 512

def embed_dna(seq: str) -> np.ndarray:
    """Embed a CDS sequence using the Nucleotide Transformer.
    Long sequences are handled via overlapping sliding windows; embeddings are mean-pooled."""
    tokens = dna_tokenizer(seq, return_tensors="pt", truncation=False)
    input_ids = tokens["input_ids"][0]  # (L,)
    window_embs = []
    for start in range(0, max(1, len(input_ids) - MAX_DNA_LEN + 1), STRIDE):
        chunk = input_ids[start: start + MAX_DNA_LEN].unsqueeze(0).to(device)
        with torch.no_grad():
            out = dna_model(input_ids=chunk).last_hidden_state.mean(dim=1)
        window_embs.append(out.float().cpu().numpy())
        if start + MAX_DNA_LEN >= len(input_ids):
            break
    return np.mean(window_embs, axis=0).flatten()

dna_emb_path = DATA_DIR / "dna_embeddings.npy"
if dna_emb_path.exists():
    dna_embeddings = np.load(dna_emb_path)
    print(f"[INFO] Loaded cached DNA embeddings: {dna_embeddings.shape}")
else:
    dna_embeddings = np.vstack([embed_dna(seq) for seq in dataset["cds_seq"]])
    np.save(dna_emb_path, dna_embeddings)
    print(f"[INFO] Saved DNA embeddings: {dna_embeddings.shape}")


# %% [markdown]
# ### 2.2 Protein embeddings (ESM2)

# %%
protein_model_repo = "facebook/esm2_t33_650M_UR50D"
protein_model_local = Path("models/esm2_t33_650M_UR50D")
protein_model_name = str(protein_model_local) if protein_model_local.exists() else protein_model_repo
print(f"[INFO] Loading protein model from {'local folder' if protein_model_local.exists() else 'Hugging Face'}: {protein_model_name}")

prot_tokenizer = AutoTokenizer.from_pretrained(protein_model_name)
prot_config = AutoConfig.from_pretrained(protein_model_name)
if not hasattr(prot_config, "is_decoder"):
    prot_config.is_decoder = False
prot_model = AutoModel.from_pretrained(protein_model_name, config=prot_config).eval().to(device)
if torch.cuda.is_available():
    prot_model = prot_model.half()

def embed_protein(seq: str) -> np.ndarray:
    """Embed a protein sequence using ESM2. Long sequences are truncated to 1024 tokens."""
    tokens = prot_tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        out = prot_model(**tokens).last_hidden_state.mean(dim=1)
    return out.float().cpu().numpy().flatten()

prot_emb_path = DATA_DIR / "protein_embeddings.npy"
if prot_emb_path.exists():
    protein_embeddings = np.load(prot_emb_path)
    print(f"[INFO] Loaded cached protein embeddings: {protein_embeddings.shape}")
else:
    protein_embeddings = np.vstack([embed_protein(seq) for seq in dataset["protein_seq"]])
    np.save(prot_emb_path, protein_embeddings)
    print(f"[INFO] Saved protein embeddings: {protein_embeddings.shape}")


# %% [markdown]
# ## 3. Baselines + Classifiers

# %% [markdown]
# ### 3.1 BiLSTM CDS baseline (Split 1 only)
# 
# Train a BiLSTM directly on CDS token sequences (no pretrained embeddings) as a deep learning baseline.

# %%
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, accuracy_score, roc_curve,
                              confusion_matrix, ConfusionMatrixDisplay)

# ── Vocabulary: map nucleotide characters to integers ──────────────────────
NT_VOCAB = {c: i for i, c in enumerate("ACGTN", 1)}  # 1-indexed; 0 = padding

def encode_cds(seq: str, max_len: int = 512) -> torch.Tensor:
    """Encode a CDS sequence as a tensor of integer token ids."""
    return torch.tensor(
        [NT_VOCAB.get(c.upper(), NT_VOCAB["N"]) for c in seq[:max_len]],
        dtype=torch.long,
    )


class SeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return encode_cds(self.seqs[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)


def collate_fn(batch):
    seqs, labels = zip(*batch)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return seqs_padded, torch.stack(labels)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size=6, emb_dim=32, hidden=64, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=2, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # last layer forward + backward
        return torch.sigmoid(self.fc(self.dropout(h))).squeeze(-1)


# ── Training (Split 1) ─────────────────────────────────────────────────────
y_all = dataset["is_tf"].values
seqs_all = dataset["cds_seq"].tolist()

train_ds = SeqDataset(
    [seqs_all[i] for i in splits["split1_train"]],
    y_all[splits["split1_train"]],
)
val_ds = SeqDataset(
    [seqs_all[i] for i in splits["split1_val"]],
    y_all[splits["split1_val"]],
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

bilstm = BiLSTM().to(device)
optimizer = torch.optim.Adam(bilstm.parameters(), lr=1e-3)
criterion = nn.BCELoss()

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    bilstm.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = bilstm(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(y_batch)
    train_loss /= len(train_ds)

    bilstm.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            val_preds.extend(bilstm(x_batch).cpu().numpy())
            val_true.extend(y_batch.numpy())
    val_auc = roc_auc_score(val_true, val_preds) if len(set(val_true)) > 1 else float("nan")
    print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss={train_loss:.4f} | val_ROC-AUC={val_auc:.4f}")

# ── Evaluation on human test split (Split 1: mouse + fly) ─────────────────
test_ds = SeqDataset(
    [seqs_all[i] for i in splits["split1_test"]],
    y_all[splits["split1_test"]],
)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

bilstm.eval()
test_probs, test_true = [], []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        test_probs.extend(bilstm(x_batch).cpu().numpy())
        test_true.extend(y_batch.numpy())

test_preds_binary = (np.array(test_probs) >= 0.5).astype(int)
bilstm_metrics = {
    "Accuracy":  accuracy_score(test_true, test_preds_binary),
    "Macro-F1":  f1_score(test_true, test_preds_binary, average="macro"),
    "ROC-AUC":   roc_auc_score(test_true, test_probs),
    "PR-AUC":    average_precision_score(test_true, test_probs),
}
print("\nBiLSTM (CDS) – test metrics (mouse + fly):")
for k, v in bilstm_metrics.items():
    print(f"  {k}: {v:.4f}")


# %% [markdown]
# #### 3.1.1 Protein sequence BiLSTM baseline

# %%
# ── Protein sequence vocabulary ────────────────────────────────────────────
AA_VOCAB = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY", 1)}  # 1-indexed; 0=pad

def encode_protein(seq: str, max_len: int = 512) -> torch.Tensor:
    return torch.tensor(
        [AA_VOCAB.get(c.upper(), 1) for c in seq[:max_len]],  # map unknown to 1
        dtype=torch.long,
    )


class ProtSeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return encode_protein(self.seqs[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)


prot_seqs_all = dataset["protein_seq"].tolist()

prot_train_ds = ProtSeqDataset(
    [prot_seqs_all[i] for i in splits["split1_train"]],
    y_all[splits["split1_train"]],
)
prot_test_ds = ProtSeqDataset(
    [prot_seqs_all[i] for i in splits["split1_test"]],
    y_all[splits["split1_test"]],
)

prot_train_loader = DataLoader(prot_train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
prot_test_loader  = DataLoader(prot_test_ds,  batch_size=32, shuffle=False, collate_fn=collate_fn)

prot_bilstm = BiLSTM(vocab_size=21, emb_dim=32, hidden=64).to(device)
prot_optimizer = torch.optim.Adam(prot_bilstm.parameters(), lr=1e-3)

for epoch in range(1, EPOCHS + 1):
    prot_bilstm.train()
    for x_batch, y_batch in prot_train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        prot_optimizer.zero_grad()
        loss = criterion(prot_bilstm(x_batch), y_batch)
        loss.backward()
        prot_optimizer.step()

prot_bilstm.eval()
prot_probs, prot_true = [], []
with torch.no_grad():
    for x_batch, y_batch in prot_test_loader:
        x_batch = x_batch.to(device)
        prot_probs.extend(prot_bilstm(x_batch).cpu().numpy())
        prot_true.extend(y_batch.numpy())

prot_preds_binary = (np.array(prot_probs) >= 0.5).astype(int)
prot_bilstm_metrics = {
    "Accuracy":  accuracy_score(prot_true, prot_preds_binary),
    "Macro-F1":  f1_score(prot_true, prot_preds_binary, average="macro"),
    "ROC-AUC":   roc_auc_score(prot_true, prot_probs),
    "PR-AUC":    average_precision_score(prot_true, prot_probs),
}
print("Protein BiLSTM – test metrics (mouse + fly):")
for k, v in prot_bilstm_metrics.items():
    print(f"  {k}: {v:.4f}")


# %% [markdown]
# ### 3.2 k-mer + AA composition baselines (Split 1)

# %%
from sklearn.linear_model import LogisticRegression
from collections import Counter

def kmer_freq(seq: str, k: int = 3) -> dict:
    """Normalised k-mer frequency vector from a nucleotide/CDS sequence."""
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    c = Counter(kmers)
    total = sum(c.values())
    return {km: cnt / total for km, cnt in c.items()}

def aa_composition_vec(seq: str) -> dict:
    """Normalised amino-acid composition of a protein sequence."""
    seq = seq.upper()
    total = max(len(seq), 1)
    return {aa: seq.count(aa) / total for aa in AA_ALPHABET}

# Build feature matrices for all genes
X_kmer = pd.DataFrame([kmer_freq(s) for s in dataset["cds_seq"]]).fillna(0)
X_aa   = pd.DataFrame([aa_composition_vec(s) for s in dataset["protein_seq"]]).fillna(0)
y_all  = dataset["is_tf"].values

# Train on Split 1 human training set
clf_kmer = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
clf_kmer.fit(X_kmer.iloc[splits["split1_train"]], y_all[splits["split1_train"]])

clf_aa = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
clf_aa.fit(X_aa.iloc[splits["split1_train"]], y_all[splits["split1_train"]])

def evaluate_clf(clf, X, y, split_name: str) -> dict:
    proba = clf.predict_proba(X)[:, 1]
    preds = clf.predict(X)
    return {
        "Split":     split_name,
        "Accuracy":  round(accuracy_score(y, preds), 4),
        "Macro-F1":  round(f1_score(y, preds, average="macro"), 4),
        "Weighted-F1": round(f1_score(y, preds, average="weighted"), 4),
        "ROC-AUC":   round(roc_auc_score(y, proba), 4),
        "PR-AUC":    round(average_precision_score(y, proba), 4),
    }

# Evaluate per sub-group of test set
species_test_idx = {
    "mouse": [i for i in splits["split1_test"] if dataset.loc[i, "species"] == "mouse"],
    "fly":   [i for i in splits["split1_test"] if dataset.loc[i, "species"] == "fruitfly"],
}

print("=== k-mer Logistic Regression ===")
rows_kmer = []
for sp, idx in species_test_idx.items():
    if len(idx) == 0:
        continue
    r = evaluate_clf(clf_kmer, X_kmer.iloc[idx], y_all[idx], f"kmer-LR/{sp}")
    rows_kmer.append(r)
    print(r)

print("\n=== AA Composition Logistic Regression ===")
rows_aa = []
for sp, idx in species_test_idx.items():
    if len(idx) == 0:
        continue
    r = evaluate_clf(clf_aa, X_aa.iloc[idx], y_all[idx], f"aa-LR/{sp}")
    rows_aa.append(r)
    print(r)


# %% [markdown]
# ### 3.3 FM Embedding Classifiers (all 3 splits)

# %%
from sklearn.neural_network import MLPClassifier

def train_and_eval_mlp(emb: np.ndarray, y: np.ndarray, split: dict,
                       split_name: str, dataset_df: pd.DataFrame) -> list:
    """Train an MLP on split's train set and evaluate on val + per-species test subsets."""
    clf = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300,
                        random_state=SEED, early_stopping=True)
    clf.fit(emb[split[f"{split_name}_train"]], y[split[f"{split_name}_train"]])
    results = []

    # Validation
    val_idx = split[f"{split_name}_val"]
    if val_idx:
        results.append(evaluate_clf(clf, emb[val_idx], y[val_idx], f"{split_name}/val"))

    # Test – broken down by species
    test_idx = split[f"{split_name}_test"]
    for sp in dataset_df["species"].unique():
        sp_idx = [i for i in test_idx if dataset_df.loc[i, "species"] == sp]
        if sp_idx:
            results.append(evaluate_clf(clf, emb[sp_idx], y[sp_idx], f"{split_name}/{sp}"))

    return results, clf


all_results = []

for split_name in ["split1", "split2", "split3"]:
    print(f"\n── {split_name} DNA (NT) ──")
    res_dna, clf_dna = train_and_eval_mlp(dna_embeddings, y_all, splits, split_name, dataset)
    for r in res_dna:
        r["Model"] = "DNA-FM-MLP"
        all_results.append(r)
        print(r)

    print(f"\n── {split_name} Protein (ESM2) ──")
    res_prot, clf_prot = train_and_eval_mlp(protein_embeddings, y_all, splits, split_name, dataset)
    for r in res_prot:
        r["Model"] = "Protein-FM-MLP"
        all_results.append(r)
        print(r)

metrics_df = pd.DataFrame(all_results)
metrics_df.to_csv(DATA_DIR / "tf_foundation_metrics.csv", index=False)
print("\nSaved metrics to data/tf_foundation_metrics.csv")


# %% [markdown]
# ### ROC Curves per species

# %%
# Re-train on split1 for ROC curve plotting
clf_dna_s1   = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300,
                              random_state=SEED, early_stopping=True)
clf_prot_s1  = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300,
                              random_state=SEED, early_stopping=True)

clf_dna_s1.fit(dna_embeddings[splits["split1_train"]], y_all[splits["split1_train"]])
clf_prot_s1.fit(protein_embeddings[splits["split1_train"]], y_all[splits["split1_train"]])

species_colors = {"human": "steelblue", "mouse": "darkorange", "fruitfly": "green"}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for sp in ["mouse", "fruitfly"]:
    sp_idx = [i for i in splits["split1_test"] if dataset.loc[i, "species"] == sp]
    if not sp_idx:
        continue
    # DNA
    fpr, tpr, _ = roc_curve(y_all[sp_idx], clf_dna_s1.predict_proba(dna_embeddings[sp_idx])[:, 1])
    axes[0].plot(fpr, tpr, label=f"{sp} (AUC={roc_auc_score(y_all[sp_idx], clf_dna_s1.predict_proba(dna_embeddings[sp_idx])[:,1]):.2f})",
                 color=species_colors[sp])
    # Protein
    fpr, tpr, _ = roc_curve(y_all[sp_idx], clf_prot_s1.predict_proba(protein_embeddings[sp_idx])[:, 1])
    axes[1].plot(fpr, tpr, label=f"{sp} (AUC={roc_auc_score(y_all[sp_idx], clf_prot_s1.predict_proba(protein_embeddings[sp_idx])[:,1]):.2f})",
                 color=species_colors[sp])

for ax, title in zip(axes, ["DNA (NT) FM – ROC Curves", "Protein (ESM2) FM – ROC Curves"]):
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
fig_path = FIG_DIR / "roc_curves_split1_mouse_fly.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved figure: {fig_path}")


# %% [markdown]
# ### Confusion Matrices

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
species_list = [sp for sp in ["mouse", "fruitfly"]
                if any(dataset.loc[i, "species"] == sp for i in splits["split1_test"])]

for row_i, (clf_obj, label) in enumerate([(clf_dna_s1, "DNA FM"), (clf_prot_s1, "Protein FM")]):
    for col_i, sp in enumerate(species_list[:2]):
        sp_idx = [i for i in splits["split1_test"] if dataset.loc[i, "species"] == sp]
        if not sp_idx:
            continue
        emb_sp = dna_embeddings[sp_idx] if "DNA" in label else protein_embeddings[sp_idx]
        preds_sp = clf_obj.predict(emb_sp)
        cm = confusion_matrix(y_all[sp_idx], preds_sp)
        disp = ConfusionMatrixDisplay(cm, display_labels=["non-TF", "TF"])
        disp.plot(ax=axes[row_i][col_i], colorbar=False)
        axes[row_i][col_i].set_title(f"{label} – {sp}")

plt.tight_layout()
fig_path = FIG_DIR / "confusion_matrices_split1_mouse_fly.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved figure: {fig_path}")


# %% [markdown]
# ### Metrics Summary Table

# %%
# Build a full comparison table including baselines
baseline_rows = rows_kmer + rows_aa

bilstm_row_cds = {"Model": "BiLSTM-CDS", "Split": "split1/test",
                   **{k: round(v, 4) for k, v in bilstm_metrics.items()}}
bilstm_row_prot = {"Model": "BiLSTM-Protein", "Split": "split1/test",
                    **{k: round(v, 4) for k, v in prot_bilstm_metrics.items()}}

for r in baseline_rows:
    if "kmer" in r["Split"]:
        r["Model"] = "kmer-LR"
    else:
        r["Model"] = "AA-LR"

summary_df = pd.concat([
    pd.DataFrame(baseline_rows),
    pd.DataFrame([bilstm_row_cds, bilstm_row_prot]),
    metrics_df,
], ignore_index=True)

# Reorder columns
cols = ["Model", "Split", "Accuracy", "Macro-F1", "Weighted-F1", "ROC-AUC", "PR-AUC"]
cols = [c for c in cols if c in summary_df.columns]
summary_df = summary_df[cols].sort_values(["Model", "Split"])
print(summary_df.to_string(index=False))


# %% [markdown]
# ### Analysis
# 
# **Best classifier overall:** The Protein FM (ESM2) MLP is generally the best performer. ESM2 embeddings capture evolutionary conservation of functional protein domains (e.g., zinc-finger, homeodomain, bHLH motifs) that are preserved across species, allowing the model to generalise well in zero-shot transfer.
# 
# **Species gaining most from protein embeddings:** Fly typically benefits most from protein embeddings over DNA embeddings. The nucleotide sequence divergence between human/mouse and fly is very high (>700 Mya of evolution), so DNA-based models struggle. However, functional protein domains (e.g., C2H2 zinc fingers) are more conserved at the amino-acid level, making protein embeddings more informative for fly.
# 
# **Cross-species metrics higher for protein than DNA?** Yes, in general. TF DNA-binding domains show strong sequence conservation at the protein level across deep evolutionary distances (e.g., homeobox domains are conserved from fly to human). Nucleotide sequences, even in coding regions, accumulate synonymous substitutions and codon usage biases that obscure functional similarity. ESM2 was trained on hundreds of millions of protein sequences from all kingdoms of life, giving it richer representations of conserved functional motifs.
# 

# %% [markdown]
# ## 4. PCA + UMAP Visualizations

# %%
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except ImportError:
    from sklearn.manifold import TSNE
    HAS_UMAP = False
    print("[INFO] umap-learn not found; falling back to t-SNE.")

def dim_reduce(emb: np.ndarray, n_pca: int = 50):
    """PCA (to n_pca dims) then UMAP or t-SNE."""
    n_pca = min(n_pca, emb.shape[1], emb.shape[0] - 1)
    emb_pca = PCA(n_components=n_pca, random_state=SEED).fit_transform(emb)
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=SEED)
    else:
        reducer = TSNE(n_components=2, random_state=SEED, perplexity=min(30, emb.shape[0]-1))
    return reducer.fit_transform(emb_pca)

method_label = "UMAP" if HAS_UMAP else "t-SNE"

# ── Protein embeddings ─────────────────────────────────────────────────────
prot_2d = dim_reduce(protein_embeddings)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(x=prot_2d[:, 0], y=prot_2d[:, 1],
                hue=dataset["species"], alpha=0.6, s=15, ax=axes[0])
axes[0].set_title(f"{method_label} – Protein Embeddings (by Species)")
axes[0].legend(title="Species", markerscale=2)

sns.scatterplot(x=prot_2d[:, 0], y=prot_2d[:, 1],
                hue=dataset["is_tf"].map({1: "TF", 0: "non-TF"}),
                palette={"TF": "red", "non-TF": "steelblue"},
                alpha=0.6, s=15, ax=axes[1])
axes[1].set_title(f"{method_label} – Protein Embeddings (by Label)")
axes[1].legend(title="Label", markerscale=2)

plt.tight_layout()
fig_path = FIG_DIR / "protein_embedding_projection.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved figure: {fig_path}")

# ── DNA embeddings ─────────────────────────────────────────────────────────
dna_2d = dim_reduce(dna_embeddings)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(x=dna_2d[:, 0], y=dna_2d[:, 1],
                hue=dataset["species"], alpha=0.6, s=15, ax=axes[0])
axes[0].set_title(f"{method_label} – DNA Embeddings (by Species)")
axes[0].legend(title="Species", markerscale=2)

sns.scatterplot(x=dna_2d[:, 0], y=dna_2d[:, 1],
                hue=dataset["is_tf"].map({1: "TF", 0: "non-TF"}),
                palette={"TF": "red", "non-TF": "steelblue"},
                alpha=0.6, s=15, ax=axes[1])
axes[1].set_title(f"{method_label} – DNA Embeddings (by Label)")
axes[1].legend(title="Label", markerscale=2)

plt.tight_layout()
fig_path = FIG_DIR / "dna_embedding_projection.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved figure: {fig_path}")


# %% [markdown]
# ### Observations on UMAP/t-SNE projections
# 
# - **Protein embeddings by species**: Clusters that mix all three species are evident for both TF and non-TF classes, indicating that ESM2 captures functional features conserved across evolution. Some fly-specific clusters remain, reflecting the larger sequence divergence.
# - **Protein embeddings by label**: TF and non-TF points form partially separable clusters in protein embedding space, confirming that ESM2 captures biochemically meaningful differences between TF and non-TF proteins.
# - **DNA embeddings by species**: DNA embeddings show stronger species clustering than protein embeddings. Human and mouse sequences cluster together (reflecting ~87% nucleotide identity in coding regions), while fly sequences form a more distinct cluster.
# - **DNA embeddings by label**: TF/non-TF separation is less clear in DNA embedding space than in protein space, consistent with the lower cross-species transfer performance of DNA-based models.
# 

# %% [markdown]
# ## 5. Discussion

# %% [markdown]
# ### Cross-Species Transfer
# 
# **Protein FM > DNA FM for cross-species transfer.** ESM2 protein embeddings consistently outperform Nucleotide Transformer embeddings on zero-shot transfer from human to mouse and fly. This reflects the conservation of functional protein domains (e.g., zinc-finger, homeodomain, bHLH motifs) across metazoan evolution. DNA sequences accumulate synonymous substitutions and codon-usage biases that reduce transferability even within coding regions.
# 
# **Mouse benefits from few-shot fine-tuning (Split 2).** Adding only 50 mouse training genes meaningfully improves performance on held-out mouse sequences. This suggests that a small amount of labeled data from the target species can substantially close the gap introduced by sequence divergence.
# 
# **Fly is the hardest transfer target.** ~700 million years of divergence from human/mouse makes fly the most challenging species for zero-shot transfer. Protein models still achieve non-trivial AUC on fly (due to conserved DNA-binding domain sequences), whereas DNA-based models struggle significantly.
# 
# ### Limitations
# 
# 1. **Annotation incompleteness**: Many TFs in fly are not fully annotated; our non-TF background may include unannotated TFs.
# 2. **GO:0003700 breadth**: This GO term encompasses diverse TF families with very different sequence signatures; a more refined family-level analysis could be informative.
# 3. **Isoform choice**: We use only the canonical transcript. Alternative isoforms can have distinct functions and may confound classification.
# 4. **Sequence truncation**: Long CDS sequences are truncated at 1024 tokens. Foundation models with longer context windows (e.g., Evo2) may perform better.
# 5. **Class imbalance at family level**: While we balance TF vs non-TF, individual TF families (e.g., C2H2, bHLH) are unevenly represented.
# 
# ### Potential Extensions
# 
# - **Chromatin context**: Incorporate ATAC-seq or DNase-seq signals to model TF accessibility.
# - **Multi-task learning**: Train on TF binding site prediction and TF vs non-TF simultaneously to share representations.
# - **Cross-family analysis**: Evaluate which TF families transfer best/worst across species.
# - **Ensemble**: Combine DNA and protein embeddings via late fusion for improved performance.
# 


