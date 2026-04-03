# Inference and Search Guide

How to run protein design with Proteina-Complexa: local execution on Linux/CUDA or macOS/MPS, SLURM cluster deployment, custom targets, and troubleshooting.

> **Documentation Map**
> - Tuning YAML configs? See [Configuration Guide](CONFIGURATION_GUIDE.md)
> - Understanding metrics? See [Evaluation Guide](EVALUATION_METRICS.md)
> - Parameter sweeps? See [Sweep System](SWEEP.md)
> - Search metadata? See [Search Metadata](SEARCH_METADATA.md)

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Design Pipeline Types](#design-pipeline-types)
3. [Configuration Architecture](#configuration-architecture)
4. [Running Locally](#running-locally)
5. [SLURM Cluster Execution](#slurm-cluster-execution)
6. [Defining Custom Targets](#defining-custom-targets)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

---

## Pipeline Overview

All design pipelines share the same four-stage structure:

| Stage | Module | Config Section | Description |
|-------|--------|----------------|-------------|
| 1. Generate | `proteinfoundation.generate` | `generation.*` | Sample structures using flow matching + reward scoring |
| 2. Filter | `proteinfoundation.filter` | `generation.filter.*` | Filter samples by reward scores |
| 3. Evaluate | `proteinfoundation.evaluate` | `metric.*` | Redesign sequences and validate with structure prediction |
| 4. Analyze | `proteinfoundation.analyze` | `aggregation.*` | Aggregate metrics, compute success rates, diversity |

---

## Design Pipeline Types

There are three main design pipelines, each targeting a different design task:

### Protein Binder Pipeline

Design protein binders for target proteins. Uses AF2 as the primary reward model and ColabDesign (AF2) or RF3 for evaluation refolding.

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=my_binder ++generation.task_name=02_PDL1
```

Apple Silicon preset:

```bash
complexa design configs/search_binder_local_pipeline_mps.yaml \
    ++run_name=my_binder_mps ++generation.task_name=02_PDL1
```

Apple Silicon + RF3 preset:

```bash
complexa design configs/search_binder_local_pipeline_mps_rf3.yaml \
    ++run_name=my_binder_mps_rf3 ++generation.task_name=02_PDL1
```

| Aspect | Setting |
|--------|---------|
| Model | Protein model (`complexa.ckpt`) |
| Generation reward | AF2 folding (TMOL, bioinformatics optional) |
| Inverse folding | SolubleMPNN |
| Evaluation folding | ColabDesign (default), RF3 |
| Evaluation type | `protein_type: binder`, `result_type: protein_binder` |
| Analysis modes | `[binder, monomer]` |

### Ligand Binder Pipeline

Design proteins that bind small-molecule ligands. Uses RF3 for both reward and evaluation since it can handle protein-ligand complexes.

```bash
complexa design configs/search_ligand_binder_local_pipeline.yaml \
    ++run_name=my_ligand_binder ++generation.task_name=39_7V11_LIGAND
```

Apple Silicon preset:

```bash
complexa design configs/search_ligand_binder_local_pipeline_mps.yaml \
    ++run_name=my_ligand_binder_mps ++generation.task_name=39_7V11_LIGAND
```

Apple Silicon + RF3 preset:

```bash
complexa design configs/search_ligand_binder_local_pipeline_mps_rf3.yaml \
    ++run_name=my_ligand_binder_mps_rf3 ++generation.task_name=39_7V11_LIGAND
```

| Aspect | Setting |
|--------|---------|
| Model | Ligand model with LoRA (`complexa_ligand.ckpt`) |
| Generation reward | RF3 folding |
| Inverse folding | LigandMPNN |
| Evaluation folding | RF3 |
| Evaluation type | `protein_type: binder`, `result_type: ligand_binder` |
| Analysis modes | `[binder, monomer]` |

### AME Pipeline (Motif + Ligand Binder)

Scaffold functional motifs with ligand context. Combines motif features (atom-spec mode) and ligand features. Uses RF3 for reward and evaluation.

```bash
complexa design configs/search_ame_local_pipeline.yaml \
    ++run_name=my_ame ++generation.task_name=M0096_1chm
```

Apple Silicon preset:

```bash
complexa design configs/search_ame_local_pipeline_mps.yaml \
    ++run_name=my_ame_mps ++generation.task_name=M0096_1chm
```

Apple Silicon + RF3 preset:

```bash
complexa design configs/search_ame_local_pipeline_mps_rf3.yaml \
    ++run_name=my_ame_mps_rf3 ++generation.task_name=M0096_1chm
```

| Aspect | Setting |
|--------|---------|
| Model | AME model with LoRA (`complexa_ame.ckpt`) |
| Generation reward | RF3 folding |
| Inverse folding | LigandMPNN |
| Evaluation folding | RF3 |
| Evaluation type | `protein_type: motif_binder`, `result_type: motif_ligand_binder` |
| Analysis modes | `[motif_binder, binder, monomer]` |
| Extra metrics | Motif RMSD, motif sequence recovery, ligand clash detection |

### Motif Binder Evaluation (Standalone)

Each base binder type (protein, ligand) has a **motif counterpart** that adds motif preservation metrics on top of the standard binder evaluation. The AME pipeline automatically uses `motif_ligand_binder`, but you can also run motif binder evaluation standalone on outputs from any binder pipeline:

```bash
# Motif protein binder evaluation (on outputs from protein binder pipeline)
complexa evaluate configs/evaluate_motif_binder.yaml \
    ++dataset.task_name=MY_MOTIF_TASK \
    ++metric.binder_folding_method=colabdesign \
    ++metric.inverse_folding_model=soluble_mpnn

# Analysis (set result_type to match)
complexa analyze configs/analyze_motif_binder.yaml \
    ++result_type=motif_protein_binder
```

| Variant | `result_type` | Binder thresholds | Motif thresholds |
|---------|---------------|-------------------|------------------|
| Motif Protein Binder | `motif_protein_binder` | i_pAE*31 <= 7.0, pLDDT >= 0.8, scRMSD_ca < 2.0 | motif_rmsd < 2.0, seq_recovery >= 1.0 |
| Motif Ligand Binder | `motif_ligand_binder` | scRMSD_bb3 <= 2.0 | motif_rmsd <= 1.5, seq_recovery >= 1.0, no ligand clashes |

---

## Configuration Architecture

The pipeline uses a modular config system. Each top-level pipeline config composes stage-specific sub-configs via Hydra defaults:

```
configs/search_binder_local_pipeline.yaml
├── pipeline/binder/binder_generate.yaml    → generation.*
├── pipeline/binder/binder_evaluate.yaml    → metric.*
└── pipeline/binder/binder_analyze.yaml     → aggregation.*

configs/search_binder_local_pipeline_mps.yaml
└── inherits search_binder_local_pipeline.yaml with MPS-safe overrides

configs/search_binder_local_pipeline_mps_rf3.yaml
└── inherits search_binder_local_pipeline_mps.yaml and enables RF3 binder metrics

configs/search_ligand_binder_local_pipeline.yaml
├── pipeline/ligand_binder/ligand_binder_generate.yaml    → generation.*
├── pipeline/ligand_binder/ligand_binder_evaluate.yaml    → metric.*
└── pipeline/ligand_binder/ligand_binder_analyze.yaml     → aggregation.*

configs/search_ligand_binder_local_pipeline_mps.yaml
└── inherits search_ligand_binder_local_pipeline.yaml with MPS-safe overrides

configs/search_ligand_binder_local_pipeline_mps_rf3.yaml
└── inherits search_ligand_binder_local_pipeline_mps.yaml and enables RF3 binder metrics

configs/search_ame_local_pipeline.yaml
├── pipeline/ame/ame_generate.yaml    → generation.*
├── pipeline/ame/ame_evaluate.yaml    → metric.*
└── pipeline/ame/ame_analyze.yaml     → aggregation.*

configs/search_ame_local_pipeline_mps.yaml
└── inherits search_ame_local_pipeline.yaml with MPS-safe overrides

configs/search_ame_local_pipeline_mps_rf3.yaml
└── inherits search_ame_local_pipeline_mps.yaml and enables RF3 motif-binder metrics
```

For the full config structure, pipeline YAML examples, and every configurable parameter, see the [Configuration Guide](CONFIGURATION_GUIDE.md).

> **Note**: You can also run individual modules directly with `python -m proteinfoundation.generate`, `python -m proteinfoundation.evaluate`, etc. The `complexa` CLI wraps these with additional validation and logging.

---

## Running Locally

### Quick Start

Linux / CUDA:

```bash
# Protein binder design
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=my_binder ++generation.task_name=02_PDL1

# Ligand binder design
complexa design configs/search_ligand_binder_local_pipeline.yaml \
    ++run_name=my_ligand_binder ++generation.task_name=39_7V11_LIGAND

# AME motif scaffolding
complexa design configs/search_ame_local_pipeline.yaml \
    ++run_name=my_ame ++generation.task_name=M0096_1chm
```

macOS / Apple Silicon (MPS presets):

```bash
# Protein binder design (MPS preset)
complexa design configs/search_binder_local_pipeline_mps.yaml \
    ++run_name=my_binder_mps ++generation.task_name=02_PDL1

# Ligand binder design (MPS preset)
complexa design configs/search_ligand_binder_local_pipeline_mps.yaml \
    ++run_name=my_ligand_binder_mps ++generation.task_name=39_7V11_LIGAND

# AME motif scaffolding (MPS preset)
complexa design configs/search_ame_local_pipeline_mps.yaml \
    ++run_name=my_ame_mps ++generation.task_name=M0096_1chm
```

macOS / Apple Silicon (MPS + RF3 presets):

```bash
# Protein binder design (MPS + RF3 preset)
complexa design configs/search_binder_local_pipeline_mps_rf3.yaml \
    ++run_name=my_binder_mps_rf3 ++generation.task_name=02_PDL1

# Ligand binder design (MPS + RF3 preset)
complexa design configs/search_ligand_binder_local_pipeline_mps_rf3.yaml \
    ++run_name=my_ligand_binder_mps_rf3 ++generation.task_name=39_7V11_LIGAND

# AME motif scaffolding (MPS + RF3 preset)
complexa design configs/search_ame_local_pipeline_mps_rf3.yaml \
    ++run_name=my_ame_mps_rf3 ++generation.task_name=M0096_1chm
```

### macOS From Scratch (Recommended)

For new users on Apple Silicon, use two separate conda environments:

- `foundry` for RF3, ProteinMPNN/LigandMPNN, and AtomWorks.
- `proteina-complexa` for running `complexa`.

Recommended split:
- Keep `foundry` on Python 3.12.
- Use Python 3.13 for the `proteina-complexa` runtime env (default path).

Install Foundry from the Apple Silicon fork: <https://github.com/fnachon/foundry>
Install Proteina-Complexa from the MPS-enabled fork: <https://github.com/fnachon/Proteina-Complexa>
Use that Proteina-Complexa fork URL as the default clone source (it contains the required fixes and patches).
Canonical values for this setup are mirrored in `pyproject.toml` under
`[tool.proteina.installation.macos]`.

```bash
# 0) Clone Proteina-Complexa (MPS fork)
git clone https://github.com/fnachon/Proteina-Complexa
cd Proteina-Complexa

# 1) Foundry (separate env)
conda create -n foundry python=3.12 -y
conda run -n foundry pip install torch
conda run -n foundry pip install "rc-foundry[all] @ git+https://github.com/fnachon/foundry.git"
conda run -n foundry foundry install base-models

# 2) Proteina-Complexa runtime env (default)
conda create -n proteina-complexa python=3.13 -y
./env/build_uv_env.sh --python 3.13
source .venv/bin/activate
conda run -n proteina-complexa pip install -e .
conda run -n proteina-complexa pip install jax-mps
conda run -n proteina-complexa complexa init uv --force
source env.sh
conda run -n proteina-complexa complexa download --complexa-all
conda install -n proteina-complexa -c conda-forge openbabel -y
conda install -n proteina-complexa -c conda-forge rdkit -y

# 3) Point Complexa to RF3 installed in foundry
export RF3_EXEC_PATH=$(conda run -n foundry which rf3)
export RF3_CKPT_PATH=$HOME/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt
```

Optional legacy runtime env on Python 3.12:

```bash
# Legacy alternate runtime env (side-by-side with the default py3.13 env)
conda create -n proteina-complexa-py312 python=3.12 -y
./env/build_uv_env.sh --python 3.12
source .venv/bin/activate
conda run -n proteina-complexa-py312 pip install -e .
conda run -n proteina-complexa-py312 pip install "jax==0.4.26" "jaxlib==0.4.26" "jax-metal==0.1.0"
# Only needed for jax-metal legacy path:
export ENABLE_PJRT_COMPATIBILITY=1
conda run -n proteina-complexa-py312 complexa init uv --force
```

Optional checkpoint wiring for ProteinMPNN/LigandMPNN:

```bash
mkdir -p community_models/ProteinMPNN/vanilla_model_weights
mkdir -p community_models/ProteinMPNN/ca_model_weights
mkdir -p community_models/ProteinMPNN/soluble_model_weights
mkdir -p community_models/LigandMPNN/model_params

ln -sf $HOME/.foundry/checkpoints/proteinmpnn_v_48_020.pt community_models/ProteinMPNN/vanilla_model_weights/v_48_020.pt
ln -sf $HOME/.foundry/checkpoints/proteinmpnn_v_48_020.pt community_models/ProteinMPNN/ca_model_weights/v_48_020.pt
ln -sf $HOME/.foundry/checkpoints/proteinmpnn_v_48_020.pt community_models/ProteinMPNN/soluble_model_weights/v_48_020.pt
ln -sf $HOME/.foundry/checkpoints/proteinmpnn_v_48_020.pt community_models/LigandMPNN/model_params/proteinmpnn_v_48_020.pt
ln -sf $HOME/.foundry/checkpoints/ligandmpnn_v_32_010_25.pt community_models/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt
```

### macOS (Apple Silicon / MPS)

Use the dedicated `*_mps.yaml` pipeline presets for local runs on Apple Silicon:

- `configs/search_binder_local_pipeline_mps.yaml`
- `configs/search_ligand_binder_local_pipeline_mps.yaml`
- `configs/search_ame_local_pipeline_mps.yaml`

Use `*_mps_rf3.yaml` when RF3 is available in your environment:

- `configs/search_binder_local_pipeline_mps_rf3.yaml`
- `configs/search_ligand_binder_local_pipeline_mps_rf3.yaml`
- `configs/search_ame_local_pipeline_mps_rf3.yaml`

These presets are optimized for MPS compatibility:

- `generation.search.algorithm=single-pass`
- `generation.reward_model=null`
- binder/motif-binder refolding metrics disabled by default
- `metric.compute_monomer_metrics=true`, `metric.compute_esm_metrics=true`
- `aggregation.analysis_modes=[monomer]`

OpenFold note for macOS:

- Proteina-Complexa imports `openfold` from the vendored `community_models/openfold` tree.
- `openfold-3` is not used by Complexa pipeline code.
- After updating Complexa, regenerate and source env setup so vendored community models are on `PYTHONPATH`:

```bash
complexa init uv --force
source env.sh
```

If you have RF3/AF2 tooling available and want full binder metrics, use the `*_mps_rf3.yaml` presets.

Keep Complexa running in your dedicated env (for example `proteina-complexa`). If RF3 is installed in a separate env (for example `foundry` from <https://github.com/fnachon/foundry>), set RF3 env vars first:

```bash
export RF3_EXEC_PATH=$(conda run -n foundry which rf3)
export RF3_CKPT_PATH=$HOME/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt
```

Then run the RF3-enabled preset:

```bash
conda run -n proteina-complexa complexa design configs/search_binder_local_pipeline_mps_rf3.yaml \
    ++run_name=my_binder_mps_rf3 \
    ++generation.task_name=02_PDL1
```

### Validate Before Running

```bash
complexa validate design configs/search_binder_local_pipeline.yaml
complexa validate design configs/search_ligand_binder_local_pipeline.yaml
complexa validate design configs/search_ame_local_pipeline.yaml

# Apple Silicon (MPS presets)
complexa validate design configs/search_binder_local_pipeline_mps.yaml
complexa validate design configs/search_ligand_binder_local_pipeline_mps.yaml
complexa validate design configs/search_ame_local_pipeline_mps.yaml

# Apple Silicon (MPS + RF3 presets)
complexa validate design configs/search_binder_local_pipeline_mps_rf3.yaml
complexa validate design configs/search_ligand_binder_local_pipeline_mps_rf3.yaml
complexa validate design configs/search_ame_local_pipeline_mps_rf3.yaml
```

### Individual Stages

```bash
complexa generate configs/search_binder_local_pipeline.yaml
complexa filter configs/search_binder_local_pipeline.yaml
complexa evaluate configs/search_binder_local_pipeline.yaml
complexa analyze configs/search_binder_local_pipeline.yaml
```

The same stage commands work for all pipeline configs -- just substitute the config path.

### CLI Options

```bash
# Verbose mode (show output instead of logging to file)
complexa design configs/search_binder_local_pipeline.yaml --verbose

# Override any config parameter with ++key=value
complexa design configs/search_binder_local_pipeline.yaml \
    ++generation.args.nsteps=200 \
    ++metric.binder_folding_method=rf3_latest
```

### Common Overrides

```bash
# Change target
++generation.task_name=33_TrkA

# Change search algorithm
++generation.search.algorithm=beam-search
++generation.search.beam_search.beam_width=8

# Change sampling steps (fewer = faster, lower quality)
++generation.args.nsteps=200

# Change folding model for evaluation
++metric.binder_folding_method=rf3_latest

# Change reward weights (protein binder -- AF2)
++generation.reward_model.reward_models.af2folding.reward_weights.i_pae=-2.0

# Change reward weights (ligand binder / AME -- RF3)
++generation.reward_model.reward_models.rf3folding.reward_weights.min_ipAE=-2.0

# Experimental refinement backend on Apple Silicon (requires Apple JAX backend)
++generation.refinement.algorithm=sequence_hallucination
++generation.refinement.jax_backend=mps

# Change success thresholds for analysis
++aggregation.success_thresholds.i_pAE.threshold=5.0

# Change filter settings
++generation.filter.filter_samples_limit=500
++generation.filter.reward_threshold=0.5
```

For the full list of configurable parameters, see the [Configuration Guide](CONFIGURATION_GUIDE.md#common-overrides-cheat-sheet).

### Quick Local Test

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=quick_test \
    ++generation.task_name=02_PDL1 \
    ++generation.args.nsteps=100 \
    ++generation.dataloader.dataset.nres.nsamples=2
```

| Config | Use case |
|--------|----------|
| `search_binder_local_pipeline.yaml` | Protein-protein binder design (local) |
| `search_binder_local_pipeline_mps.yaml` | Protein-protein binder design (Apple Silicon / MPS preset) |
| `search_binder_local_pipeline_mps_rf3.yaml` | Protein-protein binder design (Apple Silicon / MPS + RF3 preset) |
| `search_ligand_binder_local_pipeline.yaml` | Small-molecule binder design (local) |
| `search_ligand_binder_local_pipeline_mps.yaml` | Small-molecule binder design (Apple Silicon / MPS preset) |
| `search_ligand_binder_local_pipeline_mps_rf3.yaml` | Small-molecule binder design (Apple Silicon / MPS + RF3 preset) |
| `search_ame_local_pipeline.yaml` | AME motif scaffolding (local) |
| `search_ame_local_pipeline_mps.yaml` | AME motif scaffolding (Apple Silicon / MPS preset) |
| `search_ame_local_pipeline_mps_rf3.yaml` | AME motif scaffolding (Apple Silicon / MPS + RF3 preset) |
| `search_binder_pipeline.yaml` | Protein binder design (SLURM cluster) |

---

## SLURM Cluster Execution

### Setup

1. Create user configuration:

```bash
cp slurm_utils/.user_info_example slurm_utils/.user_info
```

2. Edit `.user_info` with your cluster credentials:

```bash
USER="your_username"
REMOTE="cluster.example.com"
ROOT_REMOTE="/path/to/workspace"
ENV_BINDER="complexa"
PYTHON_BINDER="/path/to/python"
MAMBA_EXEC="/path/to/mamba"
```

### Launch Scripts

**From local machine (SSH to cluster):**

```bash
# Single target
bash slurm_utils/launch_protein_binder_search_from_local_conda.sh 02_PDL1

# Single target with run number
bash slurm_utils/launch_protein_binder_search_from_local_conda.sh 02_PDL1 2

# All targets from config
bash slurm_utils/launch_protein_binder_search_from_local_conda.sh

# Multiple targets via loop
bash slurm_utils/launch_protein_binder_search_target_loop_conda.sh
```

**Directly on cluster:**

```bash
./slurm_utils/launch_protein_binder_search_from_slurm_conda.sh 02_PDL1
./slurm_utils/launch_protein_binder_search_from_slurm_conda.sh  # all targets
```

### What the Launcher Does

1. **Config generation** -- Creates per-run configs in `configs/inference_configs/` and `configs/eval_configs/`
2. **Code sync** -- Rsyncs code to cluster (local launcher only)
3. **Job submission** -- Submits SLURM array jobs for each stage
4. **Monitoring** -- Waits for each stage to complete before proceeding
5. **Result download** -- Downloads results locally (local launcher only)

### Job Parallelism

Use `gen_njobs` and `eval_njobs` to parallelize across samples:

```bash
complexa evaluate configs/search_binder_pipeline.yaml \
    ++eval_njobs=20 \
    ++job_id=$SLURM_ARRAY_TASK_ID
```

Set `eval_njobs` to match `gen_njobs` so each eval job processes one generation job's outputs.

---

## Defining Custom Targets

### Protein Targets

Add entries to `configs/targets/targets_dict.yaml`:

```yaml
target_dict_cfg:
  my_target:
    source: my_targets           # Subfolder in $DATA_PATH/target_data/
    target_filename: my_protein  # PDB filename (without .pdb)
    target_input: A1-150         # Chain and residue range
    hotspot_residues: [A45, A67, A89]
    binder_length: [60, 120]     # [min, max] binder length
    pdb_id: 1abc                 # Optional: PDB ID reference
```

Then run:

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++generation.task_name=my_target
```

### Ligand Targets

Add entries to `configs/targets/ligand_targets_dict.yaml` with additional fields:

```yaml
target_dict_cfg:
  my_ligand_target:
    source: my_targets
    target_filename: my_complex
    target_input: A1-200
    binder_length: [60, 120]
    pdb_id: 2xyz
    res_name: LIG               # Ligand residue name in PDB
    ligand_only: false           # Whether the target is ligand-only
    SMILES: "CCO"               # SMILES string for the ligand
    use_bonds_from_file: true    # Use bond topology from PDB
```

### Target Input Format

The `target_input` field specifies which residues to use:

- `A1-150` -- Chain A, residues 1-150
- `A1-100,B1-50` -- Multiple chains/ranges
- `A` -- Entire chain A

### Hotspot Residues

Interface residues the binder should contact:

```yaml
hotspot_residues: [A45, A67, A89, A102]
```

---

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'openfold'`:

- Complexa expects vendored OpenFold at `community_models/openfold`.
- Regenerate and source env setup:

```bash
complexa init uv --force
source env.sh
```

- Install Complexa into your dedicated runtime env (recommended: `proteina-complexa`):

```bash
conda run -n proteina-complexa pip install -e .
```

- If you instead see `ModuleNotFoundError: No module named 'Bio'`, install Biopython in the same runtime env:

```bash
conda run -n proteina-complexa pip install biopython
```

- If you see `ModuleNotFoundError: No module named 'rdkit'`, install RDKit in the same runtime env:

```bash
conda install -n proteina-complexa -c conda-forge rdkit -y
```

Manual fallback (if you are not using `env.sh`):

```bash
export PYTHONPATH=/path/to/project/src:/path/to/project/community_models:$PYTHONPATH
```

### Missing Model Weights

```bash
complexa download --status
```

### Config Validation Failures

```bash
complexa validate design configs/search_binder_local_pipeline.yaml --verbose
```

### RF3 Environment Variables

RF3 requires `RF3_CKPT_PATH` and `RF3_EXEC_PATH` to be set. Add them to your `.env` or export them:

```bash
export RF3_EXEC_PATH=$(conda run -n foundry which rf3)
export RF3_CKPT_PATH=$HOME/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt
```

This is the recommended macOS setup with a separate `foundry` env.

### MPS Unavailable (`torch.backends.mps.is_available() == False`)

For Apple Silicon presets, Complexa now logs backend diagnostics automatically (torch version, macOS version, MPS built/available state, and runtime reason).

Quick check in your runtime env:

```bash
conda run -n proteina-complexa python -c "import torch; print('mps_built', torch.backends.mps.is_built()); print('mps_available', torch.backends.mps.is_available())"
```

If you want to explicitly request MPS selection when available:

```bash
export COMPLEXA_ACCELERATOR=mps
```

If MPS is still unavailable, Complexa will fall back to CPU and print the exact PyTorch runtime reason in logs.

### Experimental Apple JAX for Refinement

`sequence_hallucination` refinement can be pinned to an experimental Apple JAX backend on Apple Silicon:

```bash
# Python 3.13 (default preferred path)
conda run -n proteina-complexa pip install jax-mps

++generation.refinement.algorithm=sequence_hallucination \
++generation.refinement.jax_backend=mps

# Optional legacy path on Python 3.12
conda run -n proteina-complexa pip install "jax==0.4.26" "jaxlib==0.4.26" "jax-metal==0.1.0"
# Only needed for jax-metal legacy path:
export ENABLE_PJRT_COMPATIBILITY=1
```

Supported values are `auto` (default), `mps`, `metal` (legacy alias), `gpu`, `cpu`.
`auto` prefers CUDA GPU, then Apple JAX backend (`mps`/`metal`), then CPU.
If `mps` is requested but unavailable, refinement fails fast with an explicit error.

### SLURM Job Failures

```bash
ls slurm_run_outputs/inf/   # Generation logs
ls slurm_run_outputs/eval/  # Evaluation logs
```

### Memory Issues

Reduce batch size or parallelization:

```bash
++generation.dataloader.batch_size=8
++gen_njobs=1
++eval_njobs=1
```

---

## Examples

### Production Beam Search (Protein Binder)

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=pdl1_beam_v1 \
    ++generation.task_name=02_PDL1 \
    ++generation.search.algorithm=beam-search \
    ++generation.search.beam_search.beam_width=8 \
    ++generation.search.beam_search.n_branch=4
```

### Multiple Targets on SLURM

```bash
# Launch all targets defined in config
bash slurm_utils/launch_protein_binder_search_target_loop_conda.sh

# Or specific targets
for target in 02_PDL1 33_TrkA 24_SpCas9; do
    bash slurm_utils/launch_protein_binder_search_from_local_conda.sh $target
done
```

### High-Quality Protein Binder Campaign

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=high_quality_campaign \
    ++generation.search.algorithm=beam-search \
    ++generation.search.beam_search.beam_width=8 \
    ++metric.binder_folding_method=rf3_latest \
    ++metric.num_redesign_seqs=16
```

To also enable TMOL rewards during generation, uncomment the `tmol` sub-model in `binder_generate.yaml` or add it via CLI:

```bash
++generation.reward_model.reward_models.tmol._target_=proteinfoundation.rewards.tmol_reward.TmolRewardModel \
++generation.reward_model.reward_models.tmol.enable_hbond=true \
++generation.reward_model.weights.tmol=0.5
```

### Custom Success Criteria

```bash
# Protein binder with stricter thresholds
complexa analyze configs/search_binder_local_pipeline.yaml \
    ++aggregation.success_thresholds.i_pAE.threshold=5.0 \
    ++aggregation.success_thresholds.scRMSD.threshold=1.0

# AME motif binder with custom thresholds
complexa analyze configs/search_ame_local_pipeline.yaml \
    ++aggregation.motif_binder_success_thresholds.motif_rmsd_pred.threshold=1.0 \
    ++aggregation.motif_binder_success_thresholds.motif_seq_recovery.threshold=0.8
```

### Custom Filter Settings

```bash
complexa filter configs/search_binder_local_pipeline.yaml \
    ++generation.filter.filter_samples_limit=200 \
    ++generation.filter.reward_threshold=0.3
```
