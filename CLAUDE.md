# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Capillary-flow is a Python pipeline for analyzing blood flow through capillaries above the fingernail. It processes microscopy video data through image preprocessing, segmentation, centerline detection, kymograph generation, velocity extraction, and statistical analysis across 48+ participants with various health conditions.

## Environment Setup

```bash
conda env create -f environment.yml       # Full environment (Python 3.9.13)
conda env create -f environment_slim.yml   # Minimal environment
python setup.py install                    # Install src package for cross-module imports
```

There is no formal test suite or test runner. Validation is done by running pipeline scripts and checking outputs.

## Pipeline Order

Scripts are run in this order (see `docs/pipeline_explanation.md` for details):

1. **Contrast enhancement**: `src/capillary_contrast.py` - histogram-based contrast adjustment
2. **Background generation**: `src/write_background_file.py` - median/mean of stabilized frames
3. **Segmentation**: External step using hasty.ai on background images
4. **Capillary naming**: `scripts/cap_name_pipeline2.py` -> `src/name_capillaries.py` - connected component labeling
5. **Manual review**: Researcher updates naming CSV files to assign consistent IDs across videos
6. **Capillary renaming**: `scripts/cap_rename_pipeline2.py` -> `src/rename_capillaries.py`
7. **Centerline detection**: `scripts/centerline_pipeline.py` -> `src/find_centerline.py` - skeleton extraction via FilFinder
8. **Kymograph generation**: `src/make_kymograph.py` - space-time plots along centerlines
9. **Velocity calculation**: `src/make_velocities_tyler.py` or `src/metha_velocities.py` - Hough transform line detection on kymographs
10. **Velocity validation**: `scripts/gui_kymos.py` - manual GUI-based velocity correction
11. **DataFrame compilation**: `src/analysis/df_pipeline.py` -> `src/analysis/make_big_df.py`
12. **Statistical analysis and plotting**: Various scripts in `src/analysis/`

Pipeline scripts in `scripts/` typically take a participant ID as a command-line argument (e.g., `python scripts/cap_name_pipeline2.py part09`).

## Key Constants

```python
PIX_UM = 2.44           # Pixels per micrometer (1.74 for old camera)
standard_fps = 227.8    # Frames per second (113.9 for old camera)
```

## Architecture

### Path Management

All paths are resolved through `src/config.py` using hostname-based lookup. Import paths via:
```python
from src.config import PATHS
cap_flow_path = PATHS['cap_flow']
```
Do NOT use `platform.node()` hostname checks in individual files.

### Source Structure

- **`src/`** - Core pipeline modules (contrast, segmentation, centerlines, kymographs, velocities)
- **`src/analysis/`** - Statistical analysis and visualization (86+ files). Core files:
  - `make_big_df.py` - Assembles the main DataFrame from all participants (~4700 lines)
  - `df_pipeline.py` - Orchestrates DataFrame compilation
  - `plot_big.py` - Main visualization module (~5100 lines)
  - `hysteresis.py` / `hysteresis_stats.py` - Pressure-dependent flow analysis
  - `stiffness_coeff.py` - Computes per-participant stiffness index (SI) metrics from pressure-velocity curves
  - `plot_stiffness.py` - Generates comparison plots, boxplots, age-adjusted scatterplots, and summary tables for stiffness metrics
  - `health_classifier.py` - ML-based health condition classification
  - `anova.py` - ANOVA statistical tests
  - `figs_ci.py` - Confidence interval figures
  - `shear_analysis.py` - Shear stress/rate analysis (supplement)
  - `cdf_figures.py` - CDF visualizations
  - `create_ks_tables.py` - Kolmogorov-Smirnov test tables
- **`src/tools/`** - Utility functions (parsing, plotting helpers, data loading, frog analysis)
- **`src/simulation/`** - Flow simulation code
- **`scripts/`** - Pipeline orchestration scripts (70+ files), run in defined order above
- **`frog/`** - Frog capillary study data and analysis (comparative model)

### Data Layout

```
data/part{XX}/{YYMMDD}/loc{XX}/
  ├── vids/vid{XX}/
  │   ├── moco/          # Stabilized image frames
  │   ├── mocoslice/     # Subset stabilization (some videos)
  │   └── metadata/Results.csv
  ├── segmented/hasty/   # Segmentation masks
  └── centerlines/coords/
```

Results go to `results/` with subdirectories for kymographs, velocities, stats, size, Hysteresis, Stiffness, etc.

### DataFrame Schema

The main DataFrame (`make_big_df.py`) uses these key columns:
- **Participant info**: `Participant` (part##), `Age`, `Sex`, `Birthday`
- **Health flags** (boolean): `Hypertension`, `Diabetes`, `Raynauds`, `SickleCell`, `HeartDisease`
- **Vitals**: `SYS_BP`, `DIA_BP`, `Pulse`, `BP` (string "137/70")
- **Video metadata**: `Date` (YYMMDD), `Location` (loc##), `Video` (vid##), `Pressure` (psi), `FPS`, `Finger`
- **Measurements**: `Centerline Length`, `Area`, `Diameter`, `Velocity`, `Corrected Velocity`
- **Quality flags**: `Correct` (t/f), `Zero`, `Max`, `Drop`

### Stiffness Index Metrics

Two log-transformed stiffness index (SI) metrics quantify capillary mechanical behavior from pressure-velocity curves:

1. **`SI_log1p`** = `log(1 + AUC of raw velocity)` — takes the AUC of raw velocity across pressure, then applies log1p. Compresses the distribution of total area.
   - Columns: `SI_log1p_up_04_12`, `SI_log1p_averaged_04_12`, etc.
   - Computed in `stiffness_coeff.py`, stored in `results/Stiffness/stiffness_coefficients.csv`

2. **`SI_logvel`** = `AUC of log(velocity)` — uses the `Log_Video_Median_Velocity` column as input to trapezoidal integration. Integrates log-velocity vs pressure directly, closely related to geometric-mean velocity. Penalizes near-zero (stalled) velocities heavily.
   - Columns: `SI_logvel_up_04_12`, `SI_logvel_averaged_04_12`, etc.
   - Computed in `stiffness_coeff.py`, stored in `results/Stiffness/stiffness_coefficients_log.csv`

**Naming convention**: `SI_{transform}_{method}_{range}` where:
- `transform` = `log1p` (log of AUC) or `logvel` (AUC of log velocity)
- `method` = `up` (up curve only) or `averaged` (up+down interpolated average)
- `range` = `04_12` (0.4–1.2 psi) or `02_12` (0.2–1.2 psi)

**Recommended metric**: `SI_logvel_averaged_02_12` — largest effect size (Cohen's d ~0.65), significant after age adjustment (p=0.044), biologically interpretable.

**Biological context**: Stiffer (less healthy/diabetic) capillaries maintain higher flow under external pressure because they resist collapse. Higher SI = stiffer = less healthy.

**Output files**:
- `results/Stiffness/stiffness_coefficients.csv` — raw velocity SI metrics + `SI_log1p_*` columns
- `results/Stiffness/stiffness_coefficients_log.csv` — log-velocity SI metrics with `SI_logvel_*` columns
- `results/Stiffness/stiffness_metric_comparison.csv` — side-by-side effect sizes, p-values, velocity bias
- `results/Stiffness/plots/` — group boxplots and age-adjusted scatter plots for both metrics
- `results/Stiffness/plots/stiffness_significance.csv` — full significance table

## Coding Standards

Detailed standards are in `docs/coding_standards.md`. Key points:

### Plot Styling
- Figure size: `figsize=(2.4, 2.0)`
- Font: Source Sans 3 Regular (loaded via `src.config.load_source_sans()`, with fallback)
- Font sizes: 7pt base, 6pt tick labels, 5pt legend
- Output as PDF with `pdf.fonttype: 42` for editable text
- Use `sns.set_style("whitegrid")`
- Line width: 0.5

### Import Order
1. Standard library (`os`, `platform`, `typing`)
2. Third-party (`numpy`, `pandas`, `matplotlib`, `scipy`)
3. Local (`from src.config import PATHS`, `from src.tools...`)

### Docstrings
Google-style with type hints. See `docs/coding_standards.md` for templates.

## Important Files Outside src/

- `thesis.tex` - Thesis document
- Various CSVs at root level are loaded by analysis scripts
- `metadata/` - Experiment metadata files
- `docs/pipeline_explanation.md` - Detailed pipeline documentation
- `docs/coding_standards.md` - Full coding standards reference
