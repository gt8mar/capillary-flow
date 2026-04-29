# Compression Resistance Index (CRI): Calculation and Rationale

This document describes the stiffness metric used in the paper. We chose the
**Compression Resistance Index (CRI)** as the primary stiffness metric because it
captures how strongly capillary flow resists externally applied compression.

## Definition

CRI is the area under the **averaged up/down log-velocity curve** from
**0.2 to 1.2 psi**:

```text
CRI = AUC(mean(log velocity_up, log velocity_down), pressure = 0.2-1.2 psi)
```

In the code and historical output files, this is the same metric as:

```text
SI_logvel_averaged_02_12
```

The active script also writes this value as `CRI` for readability.

## Interpretation

External pressure compresses the nailfold and reduces capillary flow. A capillary
network that maintains higher flow across the 0.2-1.2 psi pressure ramp has
greater resistance to compression.

- Higher CRI means flow remains higher under external pressure, indicating greater compression resistance.
- Lower CRI means flow falls more strongly under external pressure, indicating lower compression resistance.
- CRI is a pressure-range summary, not a single-pressure measurement.

## Calculation

The active implementation is:

```text
src/analysis/stiffness_coeff.py
```

The calculation steps are:

1. Load `summary_df_nhp_video_stats.csv`.
2. Use `Log_Video_Median_Velocity` as the velocity input.
3. For each participant, average the up-ramp and down-ramp log-velocity curves at common pressure points.
4. Keep pressures from 0.2 to 1.2 psi.
5. Integrate the averaged log-velocity curve with the trapezoidal rule.
6. Save one participant-level CRI value.

Output files:

- `results/Stiffness/cri_metrics.csv`
- `results/Stiffness/stiffness_coefficients_log.csv`

The second file is retained as a compatibility filename for older plotting and
ROC scripts that expect `SI_logvel_averaged_02_12`.

## Paper Plots

The CRI script generates only the stiffness plots used for the paper, all based
on `SI_logvel_averaged_02_12` / `CRI`.

Main outputs are written to:

```text
results/Stiffness/plots/
```

No-annotation copies for figure assembly are written to:

```text
results/Stiffness/plots/no_annotations/
```

The generated CRI plot set includes:

- CRI by diabetes or disease group.
- Age-adjusted CRI versus age by group.
- CRI by age group in controls.
- CRI across age brackets in controls.
- Blood pressure comparison plots used alongside CRI.
- CRI correlations with age and systolic/diastolic blood pressure.
- A CRI/BP condition comparison panel.

Statistical rows from these paper plots are saved to:

```text
results/Stiffness/plots/cri_paper_plot_significance.csv
```

## Significance Testing

The CRI paper plots use the existing statistical conventions:

- Group comparisons use two-sided Mann-Whitney U tests.
- Continuous CRI associations use Pearson linear regression.
- Age-adjusted group comparisons use OLS:

```text
CRI ~ Group + Age
```

The OLS group coefficient estimates the CRI difference between groups at a fixed
age. The age coefficient estimates the per-year CRI association.

### Age-adjusted diabetes comparison

For the typical CRI versus age scatterplot, the points are shown by diabetes
status and dashed trend lines are drawn separately for control and diabetic
participants. Those lines are visual aids. The reported diabetes significance is
not taken from the separate line fits; it comes from the age-adjusted OLS model:

```text
SI_logvel_averaged_02_12 ~ Group + Age
```

where `Group = 1` for diabetic participants and `Group = 0` for controls. This
tests whether diabetes status explains CRI after accounting for the linear
association between CRI and age.

Current saved results for `SI_logvel_averaged_02_12` are:

- Unadjusted diabetic versus control CRI: Mann-Whitney U, `p = 0.0218`.
- Age-adjusted diabetes effect: OLS group coefficient `+1.003`, `p = 0.0367`, `n = 72`.
- Age effect in the same OLS model: coefficient `+0.0376` CRI units/year, `p = 0.000352`.

These rows are saved in:

```text
results/Stiffness/plots/stiffness_significance.csv
```

### CRI predictor analysis

The dedicated CRI predictor analysis is implemented in:

```text
src/analysis/cri_predictor_analysis.py
```

This analysis uses controls only (`SET == "set01"`) and evaluates how age, sex,
and systolic blood pressure affect participant-level CRI. It loads:

```text
results/Stiffness/stiffness_coefficients_log.csv
```

and writes results to:

```text
results/CRI_Analysis/
```

The main ANOVA/OLS model is:

```text
CRI ~ Age + C(Sex) + SYS_BP
```

For the current complete control dataset (`n = 35`), age is the only significant
predictor:

- Age: `F = 18.35`, `p = 0.000165`, coefficient `+0.0542` CRI units/year.
- Sex: `F = 0.020`, `p = 0.888`, not significant.
- Systolic blood pressure: `F = 0.062`, `p = 0.804`, not significant.
- Model fit: `R^2 = 0.379`, adjusted `R^2 = 0.319`.

The interaction model is:

```text
CRI ~ Age + C(Sex) + SYS_BP + Age:C(Sex) + Age:SYS_BP + C(Sex):SYS_BP
```

Age remains significant in that model (`p = 0.000247`), while the interaction
terms are not significant:

- `Age:C(Sex)`: `p = 0.803`.
- `Age:SYS_BP`: `p = 0.172`.
- `C(Sex):SYS_BP`: `p = 0.888`.

Mixed-effects models are not fit for CRI in this analysis because CRI is a
participant-level summary: each participant contributes one complete CRI row.
There are no repeated CRI observations per participant for estimating a
participant random effect.

## Archived Methods

Older stiffness methods are documented separately in:

```text
docs/stiffness_methods_archive.md
```

The legacy multi-metric calculator is archived at:

```text
src/analysis/archive/stiffness_coeff_legacy.py
```

Those archived methods include raw-velocity AUC, log1p(AUC), up-only AUC,
0.4-1.2 psi ranges, stopping/restart pressure, P50, EV_lin, hysteresis,
composite stiffness, and the slope-based analytical stiffness module.
