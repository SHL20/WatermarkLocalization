# SPOT: Optimal Watermark Localization

This repository contains the simulation and real-LLM experiment code for **Optimal Watermark Localization in Mixed-Source Large Language Model Texts**. The proposed method is **SPOT** (Scanning Pivots Over Thresholds).

The repository currently provides:

- simulation notebooks for the phase-transition and heatmap experiments at $\alpha=0.5$;
- the complete $T=1$ real-LLM pipeline, from Gumbel-max watermarked text generation through editing, localization evaluation, figures, tables, and runtime summaries.

## Repository structure

```text
.
├── README.md
├── Simulations/
│   ├── phase_transition_alpha0p5.ipynb
│   └── heatmap_alpha0p5.ipynb
└── SPOT_real_llm_T1/
    ├── README.md
    ├── 01_generate_pre_edit_T1.ipynb
    ├── 02_apply_random_edits_T1.ipynb
    ├── 03_apply_adversarial_edits_T1.ipynb
    ├── 04_apply_roundtrip_translation_T1.ipynb
    ├── 05_evaluate_random_edits_SPOT_T1.ipynb
    ├── 06_evaluate_random_edits_AOL_T1.ipynb
    ├── 07_evaluate_adversarial_edits_SPOT_T1.ipynb
    ├── 08_evaluate_adversarial_edits_AOL_T1.ipynb
    ├── 09_evaluate_roundtrip_translation_SPOT_T1.ipynb
    ├── 10_evaluate_roundtrip_translation_AOL_T1.ipynb
    └── 11_summarize_T1_results.ipynb
```

## Simulations

The `Simulations/` folder contains two standalone notebooks configured for the representative case $\alpha=0.5$.

### Phase-transition simulation

`phase_transition_alpha0p5.ipynb` reproduces the one-dimensional phase-transition slices. It uses:

- Monte Carlo replication count `B = 200`;
- the original SPOT calibration procedure;
- automatic CUDA use when a GPU is available, with CPU fallback.

The notebook saves

```text
outputs_discovery_v3/phase_transition_alpha0p5.pdf
```

### Heatmap simulation

`heatmap_alpha0p5.ipynb` reproduces the two-dimensional discovery-error heatmap. It uses:

- sequence length `n_heat = 10000`;
- Monte Carlo replication count `B_heat = 500`;
- a $20\times20$ grid over the displayed $(p,q)$ region;
- automatic CUDA use when a GPU is available, with CPU fallback.

The notebook saves

```text
outputs_discovery_v3/heatmap_alpha0p5.pdf
```

### Other values of $\alpha$

Both notebooks expose

```python
alpha_vocab = 0.5
```

as a configuration value. Changing this value adapts the experiment to another vocabulary-growth exponent. In the paper, the additional phase-transition cases use $\alpha\in\{0,0.25,0.75\}$, and the additional displayed heatmaps use $\alpha\in\{0.25,0.75\}$. The $\alpha=0$ heatmap uses a different plotting grid and is not included among the displayed heatmaps.

## Real-LLM experiments at $T=1$

The `SPOT_real_llm_T1/` folder contains a file-based, restartable pipeline. Each notebook reads the ZIP output from the preceding stage and writes a new ZIP for the next stage.

The main experimental configuration is:

- language model: `facebook/opt-1.3b`;
- data: C4 `realnewslike`;
- number of documents: 1,000;
- prompt length: 50 tokens;
- continuation length: 400 tokens;
- watermark context width: 5;
- generation temperature: $T=1$;
- watermark: Gumbel-max with repeated-context masking.

### Pipeline

1. `01_generate_pre_edit_T1.ipynb` generates the pre-edit watermarked continuations.
2. Notebooks 02–04 construct random, adversarial, and roundtrip-translation edits, decode and re-tokenize the edited text, recompute verifier pivots, and construct the single token-level ground-truth label `GT`.
3. Notebooks 05–10 evaluate SPOT and AOL for the three edit families.
4. `11_summarize_T1_results.ipynb` combines the evaluation outputs, generates the paper-style Figure 8, constructs the $T=1$ entries of Tables 1 and 2, and reports runtime summaries.

See [`SPOT_real_llm_T1/README.md`](SPOT_real_llm_T1/README.md) for the full input/output flow and detailed evaluation conventions.

### Methods

The public SPOT evaluation notebooks contain:

- **SPOT-plugin**, using the Li et al. optimal-weight estimator of the surviving watermark fraction;
- **SPOT-oracle**, using the true fraction computed from `GT`.

AOL is evaluated using the same saved post-edit pivots and ground-truth labels.

For target empirical FPR $0.05$, every method first searches the bin $(0.04,0.05]$. If that bin is empty, the evaluation moves to the nearest lower nonempty bin. IoU-optimal and TPR-optimal parameters are selected separately within the chosen bin.

### Final outputs

The summary notebook produces:

```text
figure8_T1.pdf
table1_T1.csv
table2_T1.csv
pivot_runtime_by_level_T1.csv
runtime_by_level_T1.csv
runtime_summary_T1.csv
T1_summary_outputs.zip
```

`figure8_T1.pdf` is saved in the paper layout as a PDF at 300 dpi.

The runtime outputs distinguish:

- pivot recomputation time;
- localization-method inference time;
- total verification time, equal to pivot recomputation plus inference.

Only the \(T=1\) pipeline reports the detailed runtime benchmark.

## Running the notebooks

The notebooks are designed for Google Colab but can also be run in a compatible local Jupyter environment. Required packages are installed in the relevant notebooks. The main dependencies include:

```text
numpy
pandas
matplotlib
torch
transformers
datasets
accelerate
sentencepiece
sacremoses
tqdm
joblib
pybind11
setuptools
```

A GPU is strongly recommended for language-model generation, roundtrip translation, and the larger simulation runs. The localization evaluations and exact pivot reconstruction used for runtime reporting run on the CPU.

## Reproducing the experiments

For simulations, run either notebook directly.

For the real-LLM study, run the notebooks in numerical order and retain each generated ZIP file for the next stage. The pipeline is independently restartable at any stage once the required input ZIP is available.
