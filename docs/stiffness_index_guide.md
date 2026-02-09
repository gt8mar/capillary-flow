# Stiffness Index: Calculation and Rationale

This document explains how we calculate the **stiffness index** (SI) from capillary flow velocity data and why we use this approach.

---

## 1. Context: What We Are Measuring

In our experiments we apply **external pressure** to the nailfold (e.g., via an inflatable 'finger-lock') and measure **red and white blood cell velocities** in capillaries. As pressure increases, flow typically decreases and can stop; as pressure is lowered, flow restarts. The relationship between applied pressure and capillary velocity reflects microvascular mechanics.

We use the term **stiffness index** as a summary measure of this pressure–velocity behavior: it quantifies how much flow the microvasculature sustains over a given pressure range. *Higher* stiffness index values indicate *more* flow over the range (stiffer vessels, less affected by external pressure); *lower* values indicate less flow (more affected by external pressure).

---

## 2. Primary Definition: SI_AUC (Area Under the Curve)

The main stiffness index we report is **SI_AUC**: the **area under the velocity–pressure curve** over a specified pressure interval.

### 2.1 Formula

For a curve with pressure \(p\) (in psi) and velocity \(v(p)\) (in μm/s), we define:

\[
\text{SI\_AUC} = \int_{p_{\min}}^{p_{\max}} v(p)\, dp
\]

- **Units:** (μm/s) × psi → e.g. **(μm/s)·psi**
- **Integration:** Implemented with the **trapezoidal rule** over the available (pressure, velocity) points within the chosen range.

### 2.2 Why Area Under the Curve?

- **Single summary:** The AUC collapses the whole velocity–pressure relationship in a given range into one number, which is easy to compare across participants and groups (e.g., control vs diabetic).
- **Integrates flow response:** It reflects total “flow capacity” over the pressure ramp: higher AUC means velocity stays higher on average over the range; lower AUC means flow drops off more (stiffer or more pressure-sensitive behavior).
- **Robust to curve shape:** Unlike a single point (e.g., velocity at one pressure), the AUC uses all points in the range, so it is less sensitive to noise at one pressure level.
- **Interpretation:** Larger SI_AUC → more flow resilient to external pressure, consistent with stiffer vessels; smaller SI_AUC → less flow, consistent with pliable microvasculature.

So, while we are not defining stiffness as “resistance to deformation” in a strict mechanical sense, we *are* defining stiffness as the cumulative resistance of blood flow to applied external pressure.

---

## 3. Implementation Details

### 3.1 Up vs Down Curves

Pressure is typically ramped **up** (e.g., 0.2 → 1.2 psi) and then **down**. We get:

- **Up curve:** velocity vs pressure as pressure increases (often flow decreases or stops).
- **Down curve:** velocity vs pressure as pressure decreases (flow restarts).

We can compute SI_AUC in two ways:

1. **Up curve only**  
   Use only the up-ramp velocity–pressure points. This emphasizes the response as pressure is increased (flow reduction / stopping).

2. **Averaged curve**  
   For each pressure in the range, we take the mean of the up and down velocities (interpolating as needed), then integrate this averaged curve. This reduces the effect of hysteresis and gives a single curve representative of the full cycle.

Both variants are implemented in `src/analysis/stiffness_coeff.py` (e.g. `stiffness_coeff_up_*` and `stiffness_coeff_averaged_*`).

### 3.2 Pressure Ranges

Standard ranges used in the code:

- **0.4–1.2 psi**  
  - Default range for the main analyses.  
  - Avoids the very low end (0.2–0.4 psi) where behavior may be noisier or less relevant to flow cessation.

- **0.2–1.2 psi**  
  - Extended range including lower pressures.  
  - Useful when we want to capture the full ramp.

Constants in `stiffness_coeff.py`: `STIFFNESS_PRESSURE_MIN = 0.4`, `STIFFNESS_PRESSURE_MAX = 1.2`. The trapezoidal integration is performed only over points that fall inside the chosen \([p_{\min}, p_{\max}]\). If there are fewer than two points in that range, the function returns `NaN`.

### 3.3 Velocity Variable

Velocity is usually **median velocity per video** (or per pressure/up-down condition), e.g. the column `Video_Median_Velocity`. The same formulas can be applied to log-transformed velocity (e.g. `Log_Video_Median_Velocity`) if we want a log-scale SI_AUC; the interpretation is then in log-velocity units. All velocity values are cell velocities measured using kymographs (Forst et. al 2025).

---

## 4. Related Metrics (Same Pipeline)

The same pressure–velocity curves are used to compute several related quantities in `stiffness_coeff.py`:

| Metric | Description |
|--------|-------------|
| **Stopping pressure** | On the up curve, the first pressure at which velocity falls below a small threshold (e.g. 5 μm/s). |
| **Restart pressure** | On the down curve, the first pressure at which velocity rises back above that threshold. |
| **P50** | Pressure at which velocity reaches 50% of its maximum on the up curve. |
| **EV_lin** | Slope of the linear part of the pressure–velocity curve (e.g. over 0.2–0.8 psi), in (μm/s)/psi. |
| **Hysteresis** | Difference between mean velocity on the up curve and mean velocity on the down curve. |

A **composite stiffness** score can be formed as a weighted sum of velocity at 0.4 psi, velocity at 1.2 psi, and hysteresis; weights can be derived from a classifier (e.g. diabetes vs control) so the composite is tuned to group separation.

---

## 5. Log-Transformed SI_AUC

For some analyses we use a log transform to stabilize variance and improve normality:

\[
\log(\text{SI\_AUC} + 1)
\]

Implemented as `np.log1p(SI_AUC)`. This is used in regressions and group comparisons when the raw SI_AUC is skewed.

---

## 6. Alternative: Slope-Based Stiffness (Analytical Module)

In `src/analysis/analytical_stiffness.py` a different **stiffness index** is defined from a linear model:

\[
v = \beta_0 + \beta_1 \cdot p
\]

- **Stiffness index** = \(-\beta_1\) (negative slope), so that when velocity decreases with pressure, the index is positive.  
- **Compliance** = 1 / (stiffness index).

That metric is a **slope** (rate of change of velocity with pressure), not an area. It is used in a separate analytical pipeline (e.g. velocity profiles, health scores). The **paper and main group comparisons use SI_AUC** (area under the curve), not this slope-based index.

---

## 7. Evaluating Significance

We evaluate statistical significance in two ways: **unadjusted** (no covariates) and **age-adjusted** (controlling for age). Both are used in plots and in the main stiffness pipeline.

### 7.1 Unadjusted Analyses

**Group comparisons (e.g. Control vs Diabetic)**  
- **Test:** Two-sided **Mann-Whitney U** (Wilcoxon rank-sum).  
- **Usage:** Compare SI_AUC (and related metrics such as MAP, SBP, P50, EV_lin, log(SI_AUC+1)) between two groups.  
- **Rationale:** Non-parametric; does not assume normality and is robust to skew and outliers.  
- **Reported:** \(p\)-value for the null that the two groups have the same distribution.  
- **Implementation:** `scipy.stats.mannwhitneyu(control_vals, diabetic_vals, alternative='two-sided')` in `plot_stiffness.py`.

**Continuous associations (SI vs Age, MAP, SBP, etc.)**  
- **Test:** **Pearson linear regression** of SI (or log SI) on the continuous predictor.  
- **Usage:** Quantify association and test whether the slope is different from zero.  
- **Reported:** Correlation coefficient \(R\), \(p\)-value for the slope, and optionally the slope estimate.  
- **Implementation:** `scipy.stats.linregress(x, y)` in `plot_stiffness.py`.

These unadjusted tests answer: “Is there a difference between groups?” and “Is there a linear association with the predictor?” without accounting for age.

### 7.2 Age-Adjusted Analysis

Because tissue stiffness and many health outcomes change with **age**, we also test group differences **after controlling for age** so that any effect attributed to group (e.g. diabetes) is not confounded by age.

**Model**  
- Linear regression (OLS):  
  \[
  \text{SI} = \beta_0 + \beta_{\text{Group}} \cdot \text{Group} + \beta_{\text{Age}} \cdot \text{Age}
  \]  
  with **Group** coded as 0 = Control, 1 = Diabetic (or analogously for another binary group).  
- **Implementation:** `statsmodels.formula.api.ols` with formula `stiffness_col ~ Group + Age` in `stiffness_coeff.age_adjusted_analysis()`.

**What we report**  
- **Group \(p\)-value:** Tests \(H_0: \beta_{\text{Group}} = 0\). A small \(p\)-value means the group difference in SI is significant *after adjusting for age*.  
- **Age \(p\)-value:** Tests \(H_0: \beta_{\text{Age}} = 0\). Confirms whether age is a significant predictor of SI in the model.  
- **Coefficients:** \(\beta_{\text{Group}}\) (mean SI difference between groups at fixed age) and \(\beta_{\text{Age}}\) (change in SI per year).  
- **Model fit:** \(R^2\), adjusted \(R^2\), and overall \(F\)-test \(p\)-value.

**Rationale for age adjustment**  
- If diabetics tend to be older (or controls younger), a raw group difference in SI could partly reflect age.  
- Including Age in the model asks: “Is there a group difference in SI *at a given age*?”  
- Results are saved to `results/Stiffness/age_adjusted_analysis.json` and are shown on the age-adjusted comparison plots (e.g. SI vs Age by group with age-adjusted \(p\) in the figure).

**Requirements**  
- Columns: stiffness metric, group (e.g. `Diabetes`), and `Age`.  
- Rows with missing values in any of these are dropped.  
- Analysis is skipped if fewer than 10 complete cases remain.

**Interpretation of the group effect for methods**  
- For age-adjusted (group effect) rows, the **statistic** reported in the significance CSV is the **OLS regression coefficient** \(\beta_{\text{Group}}\), not a t- or F-statistic.  
- **Meaning:** \(\beta_{\text{Group}}\) is the estimated **mean difference in the outcome (SI)** between the Diabetic and Control groups **at a given age**. Group is coded 0 = Control, 1 = Diabetic, so a positive coefficient means the Diabetic group has a higher mean SI (on average, at fixed age) than the Control group; a negative coefficient means the opposite.  
- **Units:** The coefficient is in the same units as the outcome. For `stiffness_coeff_averaged_02_12_log` (log-velocity SI over 0.2–1.2 psi), the coefficient is in log(SI) units.  
- **Significance:** The **p-value** tests \(H_0: \beta_{\text{Group}} = 0\). So when you report a significant result (e.g. p = 0.045), you are reporting that the estimated group difference in SI (after adjusting for age) is statistically significant.  
- **Example methods sentence:** “We fitted an ordinary least squares (OLS) linear regression of stiffness index on group (0 = control, 1 = diabetic) and age. The group coefficient represents the estimated mean difference in stiffness index between diabetic and control participants at a given age. A significant group effect (e.g. \(\beta\) = 0.96, p = 0.045, n = 71) indicates that, after adjusting for age, the two groups differ in stiffness index.”

### 7.3 Export: Significance CSV

All significance test results (unadjusted and age-adjusted) are collected and written to a single CSV when the stiffness plotting pipeline is run.

- **File:** `results/Stiffness/plots/stiffness_significance.csv`
- **Generated by:** `src/analysis/plot_stiffness.py` (function `collect_significance_results()`), run automatically from `main()` after generating the figures.
- **Columns:**  
  - `analysis` — Short description of the test (e.g. “MAP by group (Control vs Diabetic)”, “stiffness_coeff_averaged_04_12 vs Age”, “stiffness_coeff_averaged_02_12_log age-adjusted (group effect)”).  
  - `test` — Test used: “Mann-Whitney U”, “Pearson regression”, or “OLS SI ~ Group + Age”.  
  - `statistic` — For group comparisons: Mann-Whitney U statistic. For Pearson regressions: correlation R. For age-adjusted OLS rows: the **regression coefficient** (\(\beta_{\text{Group}}\) or \(\beta_{\text{Age}}\)), i.e. the estimated group difference or age effect in SI units (see §7.2 for interpretation).  
  - `p_value` — \(p\)-value for the test.  
  - `n` — Total sample size.  
  - `n_control`, `n_diabetic` — Sample sizes per group (non-null only for group comparisons; null for regressions and age-adjusted rows).

**Contents**  
- **Group comparisons:** MAP, SBP, each SI variant (up/averaged, 0.4–1.2 and 0.2–1.2 psi), P50, EV_lin, composite stiffness, and log-SI variants (Mann-Whitney U).  
- **Continuous associations:** each SI column vs Age, vs MAP, vs SBP (Pearson regression).  
- **Age-adjusted (raw velocity):** for `stiffness_coeff_averaged_04_12`, `stiffness_coeff_averaged_02_12`, and `composite_stiffness`, one row for the group effect and one for the age effect from the OLS model SI ~ Group + Age.  
- **Age-adjusted (log velocity):** when `stiffness_coefficients_log.csv` exists and has Age/Diabetes, the same group and age effect rows are added for `stiffness_coeff_averaged_04_12_log` and `stiffness_coeff_averaged_02_12_log`, so the \(p\)-values shown on the “Age-Adjusted Group Comparison (Log Velocity, … psi)” plots are included in this CSV.

This CSV is the single place to look up every \(p\)-value and statistic reported in the stiffness plots and tables.

---

## 8. Summary

- **Stiffness index (SI_AUC)** = area under the velocity–pressure curve over a chosen pressure range (typically 0.4–1.2 psi or 0.2–1.2 psi), in (μm/s)·psi.
- **Rationale:** One number that summarizes how much flow is sustained over the pressure ramp; higher SI_AUC = more flow over the range, lower SI_AUC = more flow reduction (stiffer or more compromised response).
- **Implementation:** Trapezoidal integration in `src/analysis/stiffness_coeff.py`; variants for up-only vs averaged up/down curves and for different pressure ranges.
- **Usage:** Primary metric for comparing groups (e.g. control vs diabetic) and for associations with age, MAP, SBP; optionally log-transformed for regression. Related quantities (P50, EV_lin, stopping/restart pressure, hysteresis) are documented in the same module and used for complementary analyses.
- **Significance:** Unadjusted group comparisons use Mann-Whitney U; continuous associations use Pearson linear regression. Age-adjusted group differences use OLS with model SI ~ Group + Age; the group \(p\)-value from this model is the primary inference for a group effect after controlling for age. All significance results (including age-adjusted results for log-velocity SI when the log coefficients file exists) are exported to `results/Stiffness/plots/stiffness_significance.csv` when running `plot_stiffness.py`.
