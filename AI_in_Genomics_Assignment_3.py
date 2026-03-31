# %% [markdown]
# # Assignment 3: Cross-Species Foundation Models for Transcription Factors
# 
# #### Siva Satvik Mandapati - sm12779

# %% [markdown]
# **NOTE:** Run the converted python file attached as a batch process in HPC otherwise, there might be some errors and timeouts due to the downloads taking sometime.

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
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# %% [markdown]
# ## 1. Build the Multi-Species TF Dataset

# %% [markdown]
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
# ### 1.3.1 Retrieve promoter, CDS, and protein sequences

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
# ### 1.3.2 Build full dataset

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
# I noticed that building the dataset takes a lot of time because retrieving data from the endpoints is slow. To address this, I added a condition to load the dataset from local storage when it is available, instead of rebuilding it each time.

# %% [markdown]
# ### 1.4 Dataset Statistics + Visualizations
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
# Gene counts per species and label:
# 
# | species  | non-TF (0) | TF (1) | Total |
# |----------|------------|--------|-------|
# | fruitfly | 300        | 291    | 591   |
# | human    | 500        | 500    | 1000  |
# | mouse    | 500        | 500    | 1000  |

# %% [markdown]
# ![CDS Protein Length Distributions](figures/cds_protein_length_distributions.png)
# The above figure shows the distributions of CDS and protein lengths across the three species.
# 
# CDS Length: Most CDS sequences across all three species are relatively short, concentrated below 5,000 bp. However, fruitfly sequences exhibit significantly more extreme outliers, with some reaching nearly 50,000 bp, whereas human and mouse sequences generally remain below 20,000 bp
# 
# Protein Length: Similar to CDS lengths, protein sequences are largely clustered under 1,000–2,000 amino acids. Again, fruitfly proteins show the highest variance and extreme outliers, with some sequences approaching 20,000 aa, while human and mouse distributions are more tightly constrained

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
# ![GC Content by Species](figures/gc_content_by_species_tf_vs_nontf.png)
# The above figure compares the GC content of CDS sequences across species, stratified by TF vs non-TF genes. Transcription factors (TF, label 1) consistently show a higher median GC content compared to non-TF genes (label 0) across all three species. This trend is particularly prominent in human and fruitfly. In mouse, the GC content distribution is broader for both categories, with a noticeable presence of low-GC outliers in the non-TF group
# 
# Mean amino-acid composition per species (first 5 AAs shown):
# 
# | species  | A      | C      | D      | E      | F      |
# |----------|--------|--------|--------|--------|--------|
# | fruitfly | 0.0778 | 0.0217 | 0.0493 | 0.0605 | 0.0330 |
# | human    | 0.0832 | 0.0252 | 0.0397 | 0.0797 | 0.0315 |
# | mouse    | 0.1028 | 0.0341 | 0.0284 | 0.1128 | 0.0231 |
# 
# ![Mean Amino-Acid Composition](figures/mean_amino_acid_composition_by_species.png)
# The above figure shows the mean amino-acid composition of proteins across species. The mean amino-acid composition is highly conserved across human, mouse, and fruitfly, with Alanine (A), Leucine (L), and Serine (S) being among the most frequent. Subtle species-specific differences exist: mouse proteins show a significantly higher fraction of Glutamic Acid (E) compared to the other species, while fruitfly has slightly higher levels of Aspartic Acid (D)

# %% [markdown]
# ### 1.5 Train/Val/Test Splits

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
# Split sizes:
# > split1_train: 700 samples\
#   split1_val: 150 samples\
#   split1_test: 1591 samples\
#   split2_train: 1050 samples\
#   split2_val: 50 samples\
#   split2_test: 1491 samples\
#   split3_train: 1600 samples\
#   split3_val: 400 samples\
#   split3_test: 591 samples

# %% [markdown]
# ## 2. DNA / Protein Foundation Model Classifiers

# %% [markdown]
# ### 2.1 Nucleotide Transformer (CDS embeddings)

# %%
# Download model snapshot from Hugging Face Hub
from huggingface_hub import snapshot_download
local_dir = "./models/nucleotide-transformer-v2-500m-multi-species"
snapshot_download(repo_id="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", local_dir=local_dir)

# %%
from transformers import AutoTokenizer, AutoModel, AutoConfig

dna_model_repo = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
dna_model_local = Path("models/nucleotide-transformer-v2-500m-multi-species")
dna_model_name = str(dna_model_local) if dna_model_local.exists() else dna_model_repo
print(f"[INFO] Loading DNA model from {'local folder' if dna_model_local.exists() else 'Hugging Face'}: {dna_model_name}")

dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_name, trust_remote_code=True)
dna_config = AutoConfig.from_pretrained(dna_model_name, trust_remote_code=True)
# Patch attributes missing from older config JSONs (required by newer transformers)
if not hasattr(dna_config, "is_decoder"):
    dna_config.is_decoder = False
if not hasattr(dna_config, "add_cross_attention"):
    dna_config.add_cross_attention = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dna_model = AutoModel.from_pretrained(
    dna_model_name,
    config=dna_config,
    trust_remote_code=True,
    ignore_mismatched_sizes=True,
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
# Here, I added an option to load a locally downloaded model, since downloading the model during runtime was too time-consuming.

# %% [markdown]
# ### 2.2 Protein embeddings (ESM2)

# %%
# Download model snapshot from Hugging Face Hub
from huggingface_hub import snapshot_download
local_dir = "./models/esm2_t33_650M_UR50D"
snapshot_download(repo_id="facebook/esm2_t33_650M_UR50D", local_dir=local_dir)

# %%
protein_model_repo = "facebook/esm2_t33_650M_UR50D"
protein_model_local = Path("models/esm2_t33_650M_UR50D")
protein_model_name = str(protein_model_local) if protein_model_local.exists() else protein_model_repo
print(f"[INFO] Loading protein model from {'local folder' if protein_model_local.exists() else 'Hugging Face'}: {protein_model_name}")

prot_tokenizer = AutoTokenizer.from_pretrained(protein_model_name)
prot_config = AutoConfig.from_pretrained(protein_model_name)
if not hasattr(prot_config, "is_decoder"):
    prot_config.is_decoder = False
if not hasattr(prot_config, "add_cross_attention"):
    prot_config.add_cross_attention = False
prot_model = AutoModel.from_pretrained(protein_model_name, config=prot_config, ignore_mismatched_sizes=True).eval().to(device)
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
# ### 2.3 Training classifiers with FM embeddings

# %% [markdown]
# #### 2.3.1.1 BiLSTM CDS baseline (Split 1 only)
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
# BiLSTM (CDS) test metrics (mouse + fly):
# 
# | Metric   | Value  |
# |----------|--------|
# | Accuracy | 0.5845 |
# | Macro-F1 | 0.5832 |
# | ROC-AUC  | 0.6282 |
# | PR-AUC   | 0.6077 |

# %% [markdown]
# #### 2.3.1.2 Protein sequence BiLSTM baseline

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
# Protein BiLSTM test metrics (mouse + fly):
# 
# | Metric   | Value  |
# |----------|--------|
# | Accuracy | 0.5827 |
# | Macro-F1 | 0.5822 |
# | ROC-AUC  | 0.6099 |
# | PR-AUC   | 0.5929 |

# %% [markdown]
# ### 2.3.2 k-mer + AA composition baselines (Split 1)

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
# === k-mer Logistic Regression ===
# 
# {'Split': 'kmer-LR/mouse', 'Accuracy': 0.627, 'Macro-F1': 0.6218, 'Weighted-F1': 0.6218, 'ROC-AUC': 0.6205, 'PR-AUC': 0.5591}
# 
# {'Split': 'kmer-LR/fly', 'Accuracy': 0.6937, 'Macro-F1': 0.6931, 'Weighted-F1': 0.6934, 'ROC-AUC': 0.7599, 'PR-AUC': 0.7258}
# 
# === AA Composition Logistic Regression ===
# 
# {'Split': 'aa-LR/mouse', 'Accuracy': 0.679, 'Macro-F1': 0.6713, 'Weighted-F1': 0.6713, 'ROC-AUC': 0.7298, 'PR-AUC': 0.6667}
# 
# {'Split': 'aa-LR/fly', 'Accuracy': 0.7394, 'Macro-F1': 0.7378, 'Weighted-F1': 0.7381, 'ROC-AUC': 0.8277, 'PR-AUC': 0.7748}
# 
# 

# %% [markdown]
# ### 2.3.3 FM Embedding Classifiers (all 3 splits)

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
# ── split1 DNA (NT) ──
# 
# {'Split': 'split1/val', 'Accuracy': 0.5, 'Macro-F1': 0.4962, 'Weighted-F1': 0.4962, 'ROC-AUC': 0.5362, 'PR-AUC': 0.5415, 'Model': 'DNA-FM-MLP'}
# 
# {'Split': 'split1/mouse', 'Accuracy': 0.561, 'Macro-F1': 0.5605, 'Weighted-F1': 0.5605, 'ROC-AUC': 0.5683, 'PR-AUC': 0.546, 'Model': 'DNA-FM-MLP'}
# 
# {'Split': 'split1/fruitfly', 'Accuracy': 0.5702, 'Macro-F1': 0.5501, 'Weighted-F1': 0.5515, 'ROC-AUC': 0.6204, 'PR-AUC': 0.5685, 'Model': 'DNA-FM-MLP'}
# 
# 
# ── split1 Protein (ESM2) ──
# 
# {'Split': 'split1/val', 'Accuracy': 0.8333, 'Macro-F1': 0.8331, 'Weighted-F1': 0.8331, 'ROC-AUC': 0.9164, 'PR-AUC': 0.8965, 'Model': 'Protein-FM-MLP'}
# 
# {'Split': 'split1/mouse', 'Accuracy': 0.735, 'Macro-F1': 0.7296, 'Weighted-F1': 0.7296, 'ROC-AUC': 0.8347, 'PR-AUC': 0.8142, 'Model': 'Protein-FM-MLP'}
# 
# {'Split': 'split1/fruitfly', 'Accuracy': 0.8934, 'Macro-F1': 0.893, 'Weighted-F1': 0.8929, 'ROC-AUC': 0.951, 'PR-AUC': 0.9315, 'Model': 'Protein-FM-MLP'}
# 
# 
# ── split2 DNA (NT) ──
# 
# {'Split': 'split2/val', 'Accuracy': 0.64, 'Macro-F1': 0.6305, 'Weighted-F1': 0.6329, 'ROC-AUC': 0.6755, 'PR-AUC': 0.6303, 'Model': 'DNA-FM-MLP'}
# 
# {'Split': 'split2/mouse', 'Accuracy': 0.5478, 'Macro-F1': 0.5403, 'Weighted-F1': 0.5405, 'ROC-AUC': 0.5789, 'PR-AUC': 0.5768, 'Model': 'DNA-FM-MLP'}
# 
# {'Split': 'split2/fruitfly', 'Accuracy': 0.6091, 'Macro-F1': 0.6091, 'Weighted-F1': 0.6092, 'ROC-AUC': 0.6425, 'PR-AUC': 0.6021, 'Model': 'DNA-FM-MLP'}
# 
# 
# ── split2 Protein (ESM2) ──
# 
# {'Split': 'split2/val', 'Accuracy': 0.84, 'Macro-F1': 0.8333, 'Weighted-F1': 0.8347, 'ROC-AUC': 0.9231, 'PR-AUC': 0.9021, 'Model': 'Protein-FM-MLP'}
# 
# {'Split': 'split2/mouse', 'Accuracy': 0.7556, 'Macro-F1': 0.7488, 'Weighted-F1': 0.7489, 'ROC-AUC': 0.866, 'PR-AUC': 0.8388, 'Model': 'Protein-FM-MLP'}
# 
# {'Split': 'split2/fruitfly', 'Accuracy': 0.9289, 'Macro-F1': 0.9289, 'Weighted-F1': 0.9289, 'ROC-AUC': 0.9749, 'PR-AUC': 0.9636, 'Model': 'Protein-FM-MLP'}
# 
# 
# ── split3 DNA (NT) ──
# 
# {'Split': 'split3/val', 'Accuracy': 0.495, 'Macro-F1': 0.4834, 'Weighted-F1': 0.4834, 'ROC-AUC': 0.4939, 'PR-AUC': 0.4876, 'Model': 'DNA-FM-MLP'}
# 
# {'Split': 'split3/fruitfly', 'Accuracy': 0.5364, 'Macro-F1': 0.4493, 'Weighted-F1': 0.446, 'ROC-AUC': 0.6075, 'PR-AUC': 0.5831, 'Model': 'DNA-FM-MLP'}
# 
# 
# ── split3 Protein (ESM2) ──
# 
# {'Split': 'split3/val', 'Accuracy': 0.805, 'Macro-F1': 0.8002, 'Weighted-F1': 0.8002, 'ROC-AUC': 0.9168, 'PR-AUC': 0.9039, 'Model': 'Protein-FM-MLP'}
# 
# {'Split': 'split3/fruitfly', 'Accuracy': 0.9289, 'Macro-F1': 0.9289, 'Weighted-F1': 0.9288, 'ROC-AUC': 0.975, 'PR-AUC': 0.9653, 'Model': 'Protein-FM-MLP'}

# %% [markdown]
# ### Model Configurations (Hyperparameters)
# 
# | Category | Model | Key hyperparameters |
# |---|---|---|
# | FM (CDS) | Nucleotide Transformer + MLP (`DNA-FM-MLP`) | Embedding model: `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`; sliding-window embedding with `MAX_DNA_LEN=1024`, `STRIDE=512`; fp16 on GPU. Classifier: `MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300, random_state=SEED, early_stopping=True)` |
# | FM (Protein) | ESM2 + MLP (`Protein-FM-MLP`) | Embedding model: `facebook/esm2_t33_650M_UR50D`; truncation `max_length=1024`; fp16 on GPU. Classifier: `MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300, random_state=SEED, early_stopping=True)` |
# | Deep Learning baseline (CDS) | BiLSTM (`BiLSTM-CDS`) | Architecture: `vocab_size=6`, `emb_dim=32`, `hidden=64`, `num_layers=2`, `bidirectional=True`, `dropout=0.3`. Training: `batch_size=32`, `optimizer=Adam(lr=1e-3)`, `loss=BCELoss`, `epochs=10`, decision threshold `0.5` |
# | Deep Learning baseline (Protein) | BiLSTM (`BiLSTM-Protein`) | Architecture: `vocab_size=21`, `emb_dim=32`, `hidden=64`, `num_layers=2`, `bidirectional=True`, `dropout=0.3`. Training: `batch_size=32`, `optimizer=Adam(lr=1e-3)`, `loss=BCELoss`, `epochs=10`, decision threshold `0.5` |
# | Classical baseline (CDS) | k-mer Logistic Regression (`kmer-LR`) | Features: normalized 3-mer frequency (`k=3`). Classifier: `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)` |
# | Classical baseline (Protein) | Amino-acid composition Logistic Regression (`AA-LR`) | Features: normalized 20-AA composition. Classifier: `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)` |

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
# ![ROC Curves](figures/roc_curves_split1_mouse_fly.png)
# The above ROC curves show the performance of the DNA FM and Protein FM MLP classifiers on the mouse and fly test sets from Split 1. The Protein FM generally has higher AUCs than the DNA FM, especially for fly, indicating better generalisation across species.

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
# ![Confusion Matrices](figures/confusion_matrices_split1_mouse_fly.png)
# The above confusion matrices show the performance of the DNA FM and Protein FM MLP classifiers on the mouse and fly test sets from Split 1. The Protein FM demonstrates much higher sensitivity (true positive rate) than the DNA FM, particularly in the fruitfly test set where only 8 TFs were misclassified compared to 185 by the DNA FM. The DNA FM struggles significantly with mouse, showing a high number of both false positives (236) and false negatives (203), whereas the Protein FM drastically reduces false negatives to 62

# %% [markdown]
# ### Metrics Summary Table

# %%
# Build a full comparison table including baselines
baseline_rows = rows_kmer + rows_aa

bilstm_row_cds = {"Model": "BiLSTM-CDS(DL Baseline)", "Split": "split1/test",
                   **{k: round(v, 4) for k, v in bilstm_metrics.items()}}
bilstm_row_prot = {"Model": "BiLSTM-Protein(DL Baseline)", "Split": "split1/test",
                    **{k: round(v, 4) for k, v in prot_bilstm_metrics.items()}}

for r in baseline_rows:
    if "kmer" in r["Split"]:
        r["Model"] = "kmer-LR(Classical Baseline)"
    else:
        r["Model"] = "AA-LR(Classical Baseline)"

summary_df = pd.concat([
    pd.DataFrame(baseline_rows),
    pd.DataFrame([bilstm_row_cds, bilstm_row_prot]),
    metrics_df,
], ignore_index=True)

# Reorder columns
cols = ["Model(Family)", "Split", "Accuracy", "Macro-F1", "Weighted-F1", "ROC-AUC", "PR-AUC"]
cols = [c for c in cols if c in summary_df.columns]
summary_df = summary_df[cols].sort_values(["Model", "Split"])
print(summary_df.to_string(index=False))


# %% [markdown]
# | Model(Family) | Split | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |
# |---|---|---:|---:|---:|---:|---:|
# | kmer-LR(Classical Baseline) | kmer-LR/fly | 0.6937 | 0.6931 | 0.6934 | 0.7599 | 0.7258 |
# | kmer-LR(Classical Baseline) | kmer-LR/mouse | 0.6270 | 0.6218 | 0.6218 | 0.6205 | 0.5591 |
# | BiLSTM-CDS(DL Baseline) | split1/test | 0.5845 | 0.5832 | NaN | 0.6282 | 0.6077 |
# | DNA-FM-MLP | split1/fruitfly | 0.5702 | 0.5501 | 0.5515 | 0.6204 | 0.5685 |
# | DNA-FM-MLP | split1/mouse | 0.5610 | 0.5605 | 0.5605 | 0.5683 | 0.5460 |
# | DNA-FM-MLP | split1/val | 0.5000 | 0.4962 | 0.4962 | 0.5362 | 0.5415 |
# | DNA-FM-MLP | split2/fruitfly | 0.6091 | 0.6091 | 0.6092 | 0.6425 | 0.6021 |
# | DNA-FM-MLP | split2/mouse | 0.5478 | 0.5403 | 0.5405 | 0.5789 | 0.5768 |
# | DNA-FM-MLP | split2/val | 0.6400 | 0.6305 | 0.6329 | 0.6755 | 0.6303 |
# | DNA-FM-MLP | split3/fruitfly | 0.5364 | 0.4493 | 0.4460 | 0.6075 | 0.5831 |
# | DNA-FM-MLP | split3/val | 0.4950 | 0.4834 | 0.4834 | 0.4939 | 0.4876 |
# | AA-LR(Classical Baseline) | aa-LR/fly | 0.7394 | 0.7378 | 0.7381 | 0.8277 | 0.7748 |
# | AA-LR(Classical Baseline) | aa-LR/mouse | 0.6790 | 0.6713 | 0.6713 | 0.7298 | 0.6667 |
# | BiLSTM-Protein(DL Baseline) | split1/test | 0.6631 | 0.6596 | NaN | 0.7116 | 0.6584 |
# | Protein-FM-MLP | split1/fruitfly | 0.8934 | 0.8930 | 0.8929 | 0.9510 | 0.9315 |
# | Protein-FM-MLP | split1/mouse | 0.7350 | 0.7296 | 0.7296 | 0.8347 | 0.8142 |
# | Protein-FM-MLP | split1/val | 0.8333 | 0.8331 | 0.8331 | 0.9164 | 0.8965 |
# | Protein-FM-MLP | split2/fruitfly | 0.9289 | 0.9289 | 0.9289 | 0.9749 | 0.9636 |
# | Protein-FM-MLP | split2/mouse | 0.7556 | 0.7488 | 0.7489 | 0.8660 | 0.8388 |
# | Protein-FM-MLP | split2/val | 0.8400 | 0.8333 | 0.8347 | 0.9231 | 0.9021 |
# | Protein-FM-MLP | split3/fruitfly | 0.9289 | 0.9289 | 0.9288 | 0.9750 | 0.9653 |
# | Protein-FM-MLP | split3/val | 0.8050 | 0.8002 | 0.8002 | 0.9168 | 0.9039 |

# %% [markdown]
# ### Analysis
# 
# **Best classifier overall:** The Protein FM (ESM2) MLP is generally the best performer. ESM2 embeddings capture evolutionary conservation of functional protein domains (e.g., zinc-finger, homeodomain, bHLH motifs) that are preserved across species, allowing the model to generalise well in zero-shot transfer.
# 
# **Species gaining most from protein embeddings:** Fly typically benefits most from protein embeddings over DNA embeddings. The nucleotide sequence divergence between human/mouse and fly is very high (>700 Million years of evolution), so DNA-based models struggle. However, functional protein domains (e.g., C2H2 zinc fingers) are more conserved at the amino-acid level, making protein embeddings more informative for fly.
# 
# **Cross-species metrics higher for protein than DNA?** Yes, in general. TF DNA-binding domains show strong sequence conservation at the protein level across deep evolutionary distances (e.g., homeobox domains are conserved from fly to human). Nucleotide sequences, even in coding regions, accumulate synonymous substitutions and codon usage biases that obscure functional similarity. ESM2 was trained on hundreds of millions of protein sequences from all kingdoms of life, giving it richer representations of conserved functional motifs.
# 

# %% [markdown]
# ### PCA + UMAP Visualizations

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
# ![Protein Embeddings](figures/protein_embedding_projection.png)
# The above UMAP projections of ESM2 protein embeddings show that sequences partially cluster by species and mostly cluster by TF/non-TF label (stronger separation). Human and mouse sequences intermingle, reflecting their close evolutionary relationship, while fly sequences form a bit more distinct clusters close to human and mouse sequences. TFs tend to cluster together, likely due to shared functional domains, but there is also some mixing with non-TFs, indicating that the embeddings capture a spectrum of functional features rather than a strict binary separation.
# 
# 
# ![DNA Embeddings](figures/dna_embedding_projection.png)
# The UMAP projections of Nucleotide Transformer DNA embeddings show similar species clustering like the protein embeddings. Human and mouse sequences cluster together at a spot, but they also custer together with fly sequences. The separation between TF and non-TF sequences is less clear in DNA embedding space than in protein space, consistent with the lower cross-species transfer performance of DNA-based models. This suggests that while the DNA FM captures some functional signals, it is more influenced by species-specific sequence features that do not generalise as well across distant species.

# %% [markdown]
# ## 3. Discussion
# 
# 
# 1. **Cross-Species Trends and Model Performance:** The experimental results demonstrate that the Protein FM (ESM2) MLP is the best classifier overall. It significantly outperforms the DNA-based foundation model and the BiLSTM baselines, particularly in zero-shot cross-species transfer tasks. The Protein FM achieves a high ROC-AUC (e.g., 0.95 for fruitfly in Split 1) compared to the DNA FM (0.62 for fruitfly), confirming that cross-species metrics are substantially higher for protein than DNA.
# 
# 2. **Biological Interpretation:** The superior performance of protein embeddings, especially for evolutionary distant species like the fruitfly, which gains the most from this modality, is likely due to the high conservation of functional protein domains. While human and fly diverged over 700 million years ago, leading to high nucleotide sequence divergence and distinct codon usage biases, the DNA-binding domains (e.g., homeobox or zinc-finger motifs) remain highly conserved at the amino-acid level. The UMAP visualizations support this, showing that while DNA embeddings are heavily influenced by species-specific sequence features, protein embeddings form more distinct clusters based on functional labels (TF vs. non-TF) across different species.
# 
# 3. **Limitations:** There are several limitations to this study:
# 
#     1. **Annotation Bias:** The dataset assumes that all TFs are correctly annotated in Gene Ontology, meaning unannotated TFs might be mislabeled as negatives in the non-TF set.
#     2. **Missing Context:** The current models only use the primary sequence (CDS or protein) and do not account for chromatin context, evidence levels for annotations, or structural factors like isoforms.
#     3. **TF Families:** The analysis does not yet break down performance by specific TF families, which may have varying levels of evolutionary conservation.
# 
# 4. **Potential Extensions:** Future work could enhance these models by:
# 
#     1. **Multi-task Learning:** Training the model to predict multiple functional attributes simultaneously to improve the richness of the learned representations.
#     2. **Incorporating Epigenomics:** Adding features such as chromatin accessibility (ATAC-seq) or histone modifications to better predict TF activity in specific cellular contexts.
#     3. **Fine-tuning:** Performing more extensive supervised fine-tuning on diverse species to see if DNA models can overcome species-specific biases with more data


