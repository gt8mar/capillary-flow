import os
import numpy as np
import pandas as pd
from typing import List, Dict

# Import PATHS helper from config
from src.config import PATHS


def _load_merged_dataframe() -> pd.DataFrame:
    """Load summary dataframe and merge height/weight supplement.

    Returns
    -------
    pd.DataFrame
        Combined dataframe with Height and Weight columns consolidated.
    """
    data_fp = os.path.join(PATHS["cap_flow"], "summary_df_nhp_video_stats.csv")
    df = pd.read_csv(data_fp)

    # Merge height/weight supplement if available
    hw_fp = os.path.join(PATHS["cap_flow"], "height_weight.csv")
    if os.path.exists(hw_fp):
        hw = pd.read_csv(hw_fp)
        df = pd.merge(df, hw, on="Participant", how="outer")
        # Consolidate columns
        for col in ["Height", "Weight"]:
            x_col, y_col = f"{col}_x", f"{col}_y"
            if x_col in df.columns and y_col in df.columns:
                df[col] = df[x_col].fillna(df[y_col])
                df.drop(columns=[x_col, y_col], inplace=True)

    # Clean Height/Weight
    df = df.dropna(subset=["Height", "Weight"])  # keep only rows with both
    
    # Remove participants with missing SET values (e.g., part24 - excluded for bad video quality)
    print(f"Participants before SET filtering: {df['Participant'].nunique()}")
    excluded_participants = df[df["SET"].isna()]["Participant"].unique()
    df = df.dropna(subset=["SET"])
    print(f"Participants after SET filtering: {df['Participant'].nunique()}")
    
    if len(excluded_participants) > 0:
        print(f"Excluded participants: {list(excluded_participants)}")

    # Ensure numeric types
    num_cols = ["Age", "Height", "Weight", "SYS_BP", "DIA_BP", "Pressure", "Video_Median_Velocity"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _compute_baseline_velocities(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-participant baseline velocities at 0.2 psi and extrapolated 0.0 psi intercept.

    Uses the same naming convention as health_classifier.py: velocity_at_{pressure}psi
    """
    # Create participant-level dataframe
    participant_df = df.drop_duplicates("Participant").set_index("Participant")
    
    # Basic velocity statistics for each pressure (following health_classifier.py approach)
    pressure_stats = df.pivot_table(
        index='Participant',
        columns='Pressure',
        values='Video_Median_Velocity',
        aggfunc=['mean']  # Mean velocity at each pressure
    ).fillna(0)  # Fill NaN with 0 for missing pressures
    
    # Flatten multi-index columns
    pressure_stats.columns = [f'velocity_at_{pressure}psi' 
                            for (_, pressure) in pressure_stats.columns]
    
    # Add pressure-specific velocity features to participant dataframe
    for col in pressure_stats.columns:
        participant_df[col] = pressure_stats[col]
    
    # Linear fit per participant for 0 psi intercept estimation
    mean_vel = df.groupby(["Participant", "Pressure"])["Video_Median_Velocity"].mean().reset_index()
    v00 = {}
    for pid, sub in mean_vel.groupby("Participant"):
        if len(sub) < 2:
            v00[pid] = np.nan  # cannot fit line
            continue
        # simple linear regression: velocity = slope*pressure + intercept
        slope, intercept = np.polyfit(sub["Pressure"], sub["Video_Median_Velocity"], 1)
        v00[pid] = intercept  # intercept at pressure == 0

    participant_df["velocity_at_0psi"] = pd.Series(v00)

    return participant_df.reset_index()


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add BMI and tidy boolean categorical columns."""
    # BMI in kg/m^2 (Height in inches, Weight appears to be in pounds based on typical values ~150-170)
    height_m = df["Height"] * 0.0254  # Convert inches to meters
    weight_kg = df["Weight"] * 0.453592  # Convert pounds to kg
    df["BMI"] = weight_kg / (height_m ** 2)

    # Clean categorical/boolean fields
    df["Hypertension"] = df["Hypertension"].apply(lambda x: str(x).upper() == "TRUE")
    df["Diabetes"] = df["Diabetes"].apply(lambda x: str(x).upper() == "TRUE")

    return df


def _summary_table(df: pd.DataFrame, group_col: str, group_order: List[str] = None) -> pd.DataFrame:
    """Create a demographic summary table for a specified grouping column.

    Parameters
    ----------
    df : pd.DataFrame
        Participant-level dataframe with derived metrics.
    group_col : str
        Column name to group by (e.g., 'SET', 'Hypertension', 'Diabetes').
    group_order : list, optional
        Custom ordering for the columns.

    Returns
    -------
    pd.DataFrame
        Summary table with rows as variables and columns as groups (+ Overall).
    """
    cont_vars = [
        ("Age", "years"),
        ("Height", "inches"),
        ("Weight", "lbs"),
        ("BMI", "kg/m^2"),
        ("SYS_BP", "mmHg"),
        ("DIA_BP", "mmHg"),
    ]
    
    # Add velocity columns dynamically (following health_classifier.py naming)
    velocity_cols = [col for col in df.columns if col.startswith('velocity_at_')]
    # Sort to ensure consistent order (0psi first, then by pressure)
    velocity_cols = sorted(velocity_cols, key=lambda x: float(x.replace('velocity_at_', '').replace('psi', '')))
    
    for col in velocity_cols:
        cont_vars.append((col, "mm/s"))
    cat_vars = [
        ("Sex", ["M", "F"]),
        ("Hypertension", [True]),
        ("Diabetes", [True]),
    ]

    groups = list(df[group_col].dropna().unique())
    if group_order:
        groups = [g for g in group_order if g in groups]
    groups.append("Overall")

    table = {}

    # Continuous variables: mean ± SD (min-max)
    for var, unit in cont_vars:
        values = []
        for g in groups:
            sub = df if g == "Overall" else df[df[group_col] == g]
            if sub.empty or sub[var].dropna().empty:
                values.append("–")
                continue
            mean = sub[var].mean()
            sd = sub[var].std()
            # also show range
            rng = (sub[var].min(), sub[var].max())
            values.append(f"{mean:.1f} ± {sd:.1f} ({rng[0]:.1f}–{rng[1]:.1f})")
        
        # Add "calculated" label for extrapolated 0 psi velocity
        if var == "velocity_at_0psi":
            table[f"{var} (calculated, {unit})"] = values
        else:
            table[f"{var} ({unit})"] = values

    # Categorical variables: n (%) for each category of interest
    for var, cats in cat_vars:
        for cat in cats:
            row_name = f"{var} = {cat}" if var != "Sex" else ("Female" if cat == "F" else "Male")
            values = []
            for g in groups:
                sub = df if g == "Overall" else df[df[group_col] == g]
                if sub.empty:
                    values.append("–")
                    continue
                n = (sub[var] == cat).sum()
                pct = n / len(sub) * 100 if len(sub) > 0 else 0
                values.append(f"{n} ({pct:.0f}%)")
            table[row_name] = values

    # Sample size row (N)
    n_values = []
    for g in groups:
        sub = df if g == "Overall" else df[df[group_col] == g]
        n_values.append(str(len(sub)))
    table_rows = {"N": n_values}
    table_rows.update(table)

    out_df = pd.DataFrame(table_rows, index=groups).T  # rows become index
    return out_df


def _save_latex_table_with_note(table: pd.DataFrame, filepath: str, note: str) -> None:
    """Save LaTeX table with custom note appended."""
    # Generate basic LaTeX
    latex_content = table.to_latex(na_rep="–")
    
    # Add note before the closing table environment if note exists
    if note:
        # Find the last \end{tabular} and add note before \end{table}
        lines = latex_content.split('\n')
        
        # Insert note before the last few lines
        insert_idx = -3  # Usually before \end{table}
        for i in range(len(lines)-1, -1, -1):
            if '\\end{table}' in lines[i]:
                insert_idx = i
                break
        
        # Insert the note
        note_lines = [
            '\\\\',
            '\\multicolumn{' + str(len(table.columns)) + '}{l}{\\footnotesize ' + note + '}',
        ]
        
        for j, note_line in enumerate(note_lines):
            lines.insert(insert_idx + j, note_line)
        
        latex_content = '\n'.join(lines)
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(latex_content)


def create_demographic_tables() -> Dict[str, pd.DataFrame]:
    """Main driver to create and save the requested demographic tables.
    
    Creates tables with velocity columns following health_classifier.py naming:
    - velocity_at_0psi (extrapolated intercept)
    - velocity_at_0.2psi, velocity_at_0.4psi, etc. (measured at each pressure)
    """
    df_raw = _load_merged_dataframe()
    participant_df = _compute_baseline_velocities(df_raw)
    participant_df = _add_derived_columns(participant_df)
    
    # Check for excluded participants to add to table notes
    original_data = pd.read_csv(os.path.join(PATHS["cap_flow"], "summary_df_nhp_video_stats.csv"))
    hw_data = pd.read_csv(os.path.join(PATHS["cap_flow"], "height_weight.csv"))
    merged_original = pd.merge(original_data, hw_data, on="Participant", how="outer")
    for col in ["Height", "Weight"]:
        x_col, y_col = f"{col}_x", f"{col}_y"
        if x_col in merged_original.columns and y_col in merged_original.columns:
            merged_original[col] = merged_original[x_col].fillna(merged_original[y_col])
    
    # Find excluded participants (those with height/weight but missing SET)
    with_hw = merged_original.dropna(subset=["Height", "Weight"])
    excluded_participants = with_hw[with_hw["SET"].isna()]["Participant"].unique()
    
    exclusion_note = ""
    if len(excluded_participants) > 0:
        exclusion_note = f"Note: {len(excluded_participants)} participant(s) excluded for bad video quality: {', '.join(excluded_participants)}"

    # Output directory
    out_dir = os.path.join(PATHS["cap_flow"], "results", "demographics")
    os.makedirs(out_dir, exist_ok=True)

    tables = {}

    # 1. General table by SET
    set_order = ["set01", "set02", "set03"]
    table_all = _summary_table(participant_df, "SET", set_order)
    
    # Add exclusion note to CSV
    if exclusion_note:
        table_all_with_note = table_all.copy()
        # Add empty row then note
        table_all_with_note.loc[""] = [""] * len(table_all.columns)
        table_all_with_note.loc[exclusion_note] = [""] * len(table_all.columns)
        table_all_with_note.to_csv(os.path.join(out_dir, "demographics_by_SET.csv"))
    else:
        table_all.to_csv(os.path.join(out_dir, "demographics_by_SET.csv"))
    
    # Save LaTeX with custom note
    _save_latex_table_with_note(table_all, os.path.join(out_dir, "demographics_by_SET.tex"), exclusion_note)
    tables["SET"] = table_all

    # 2. Hypertension table
    table_htn = _summary_table(participant_df, "Hypertension", [False, True])
    
    # Add exclusion note to CSV
    if exclusion_note:
        table_htn_with_note = table_htn.copy()
        table_htn_with_note.loc[""] = [""] * len(table_htn.columns)
        table_htn_with_note.loc[exclusion_note] = [""] * len(table_htn.columns)
        table_htn_with_note.to_csv(os.path.join(out_dir, "demographics_by_hypertension.csv"))
    else:
        table_htn.to_csv(os.path.join(out_dir, "demographics_by_hypertension.csv"))
    
    _save_latex_table_with_note(table_htn, os.path.join(out_dir, "demographics_by_hypertension.tex"), exclusion_note)
    tables["Hypertension"] = table_htn

    # 3. Diabetes table
    table_dm = _summary_table(participant_df, "Diabetes", [False, True])
    
    # Add exclusion note to CSV
    if exclusion_note:
        table_dm_with_note = table_dm.copy()
        table_dm_with_note.loc[""] = [""] * len(table_dm.columns)
        table_dm_with_note.loc[exclusion_note] = [""] * len(table_dm.columns)
        table_dm_with_note.to_csv(os.path.join(out_dir, "demographics_by_diabetes.csv"))
    else:
        table_dm.to_csv(os.path.join(out_dir, "demographics_by_diabetes.csv"))
    
    _save_latex_table_with_note(table_dm, os.path.join(out_dir, "demographics_by_diabetes.tex"), exclusion_note)
    tables["Diabetes"] = table_dm

    print("Demographic tables saved to:", out_dir)
    return tables


if __name__ == "__main__":
    create_demographic_tables() 