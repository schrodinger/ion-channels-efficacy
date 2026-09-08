# Generated from Analysis-TRPML1.ipynb.
# The working directory is set to this script's directory so ../ paths resolve to the repository root.
from pathlib import Path as _NotebookPath
import os as _notebook_os
_notebook_os.chdir(_NotebookPath(__file__).resolve().parent)

try:
    from IPython.display import display
except ImportError:
    display = print

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from analysis_utils import combine_dataframes, ddg_barplot


# ======================
# Load ABFEP Results
# ======================


def clean_unstable_ligands(df):
    ligand_ids = df['ligand'].str.split(' ').str[0]
    unstable_mask = df.eq('unstable in binding site').any(axis=1)
    unstable_ligands = set(ligand_ids[unstable_mask])
    if not unstable_ligands:
        return df

    df_clean = df.copy()
    rows_to_fix = ligand_ids.isin(unstable_ligands)
    df_clean.loc[rows_to_fix] = (
        df_clean.loc[rows_to_fix]
        .replace('unstable in binding site', 0.0)
        .fillna(0.0)
    )
    return df_clean


# Load data for each state and lipid condition.
trpml1_a_removed_exp_lipid = pd.read_csv('../results/trpml1-a-removed-exp-lipid.csv')
trpml1_a_adapted_exp_lipid = pd.read_csv('../results/trpml1-a-adapted-exp-lipid.csv')
trpml1_i_removed_exp_lipid = pd.read_csv('../results/trpml1-i-removed-exp-lipid.csv')
trpml1_i_adapted_exp_lipid = pd.read_csv('../results/trpml1-i-adapted-exp-lipid.csv')
# Combine lipid conditions before applying the existing state-level cleanup.
trpml1_a = clean_unstable_ligands(pd.concat([
    trpml1_a_removed_exp_lipid,
    trpml1_a_adapted_exp_lipid,
], ignore_index=True))
trpml1_i = clean_unstable_ligands(pd.concat([
    trpml1_i_removed_exp_lipid,
    trpml1_i_adapted_exp_lipid,
], ignore_index=True))
# Combine active and inactive data.
df = combine_dataframes(trpml1_a, trpml1_i)
# Sort the data.
df = df.sort_values('DDG')[['Ligand_Name','DGA','DGA_Error','DGI','DGI_Error','DDG','DDG_Error']]
# Show the combined and sorted data.
display(df)


# ================================
# Experimental Data and Labels
# ================================


# Load TRPML1 labels from CSV file
labels_data = pd.read_csv('../labels/trpml1_labels.csv')
labels = labels_data.set_index('ligand')['label'].to_dict()

# Load TRPML1 electrophysiology results
trpml1_electrophysiology = pd.read_csv('../labels/trpml1_electrophysiology.csv')
trpml1_electrophysiology['Compound'] = [f'compound-{cpd}' for cpd in trpml1_electrophysiology['Compound']]

# Get binding free energies from EC50 and IC50 values
def _parse_potency_um(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s in {'', '-', 'nan', 'NaN'}:
        return np.nan
    s = s.replace('>', '').replace('<', '').replace('=', '').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

R_KCAL_PER_MOL_K = 1.98720425864083e-3
TEMPERATURE_K = 298.15

trpml1_electrophysiology['agonist_ec50_uM'] = trpml1_electrophysiology['Agonist EC50 (μM)'].apply(_parse_potency_um)
trpml1_electrophysiology['antagonist_ic50_uM'] = trpml1_electrophysiology['Antagonist IC50 (μM)'].apply(_parse_potency_um)

trpml1_electrophysiology['potency_uM_min'] = trpml1_electrophysiology[[
    'agonist_ec50_uM', 'antagonist_ic50_uM'
 ]].min(axis=1, skipna=True)

trpml1_electrophysiology['potency_source'] = np.select(
    [
        trpml1_electrophysiology['agonist_ec50_uM'] < trpml1_electrophysiology['antagonist_ic50_uM'],
        trpml1_electrophysiology['antagonist_ic50_uM'] < trpml1_electrophysiology['agonist_ec50_uM'],
        trpml1_electrophysiology['agonist_ec50_uM'].notna() & trpml1_electrophysiology['antagonist_ic50_uM'].notna(),
        trpml1_electrophysiology['agonist_ec50_uM'].notna(),
        trpml1_electrophysiology['antagonist_ic50_uM'].notna(),
    ],
    [
        'agonist_ec50',
        'antagonist_ic50',
        'both_equal',
        'agonist_ec50',
        'antagonist_ic50',
    ],
    default='none',
)

potency_m = trpml1_electrophysiology['potency_uM_min'] * 1e-6
trpml1_electrophysiology['binding_dG_kcal_mol'] = R_KCAL_PER_MOL_K * TEMPERATURE_K * np.log(potency_m)

trpml1_electrophysiology['emax'] = trpml1_electrophysiology['Agonist Emax (%)'].apply(lambda x: 0 if x=='-' else x)

display(trpml1_electrophysiology)


experimental_emax_dict = dict(zip(trpml1_electrophysiology['Compound'], trpml1_electrophysiology['emax']))
experimental_dg_dict = dict(zip(trpml1_electrophysiology['Compound'], trpml1_electrophysiology['binding_dG_kcal_mol']))


# ============
# Barplot
# ============


bar_ddg = df[df['Ligand_Name'].isin([lig for lig, label in labels.items() if label in ['agonist','antagonist']])]


_,_ = ddg_barplot(bar_ddg, labels, figsize=(4, 3), xlabel_mode='none', ylim=[-9,7.0], file_name='../plots/trpml1_ddg_barplot_nolabels')
_,_ = ddg_barplot(bar_ddg, labels, figsize=(4, 4), xlabel_mode='dataframe', ylim=[-9,7.0], file_name='../plots/trpml1_ddg_barplot')


# ================================================
# Maximum Response over Free Energy Difference
# ================================================


# Exclude ligands that are reported as >30 uM in BOTH assays from fitting.
agonist_gt30 = trpml1_electrophysiology['Agonist EC50 (μM)'].astype(str).str.strip().str.startswith('>30')
antagonist_gt30 = trpml1_electrophysiology['Antagonist IC50 (μM)'].astype(str).str.strip().str.startswith('>30')
excluded_ligands = set(trpml1_electrophysiology.loc[agonist_gt30 & antagonist_gt30, 'Compound'])
print('Excluded from Emax fit (EC50 and IC50 both >30 uM):', sorted(excluded_ligands))


def plot_max_response_over_ddg(
    experimental_emax_dict, excluded_ligands,
    color_by_class=False, draw_excluded_ligands=True,
    draw_legend=True, difference_label=False,
    xlim=[-11, 6], figsize=(4.8, 3.1), dpi=300,
    save_as=None,
):
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    all_respo = []
    all_ddg = []
    all_ddg_sigma = []

    def ligand_natural_key(lig):
        token = str(lig).split('-', 1)[-1].lower()
        index = 0
        while index < len(token) and token[index].isdigit():
            index += 1
        number = int(token[:index]) if index else float('inf')
        return number, token[index:], token

    excluded_symbols = ['P', 'X', 'D']
    excluded_sorted = sorted(excluded_ligands, key=ligand_natural_key)
    excluded_marker_map = {
        ligand: excluded_symbols[index % len(excluded_symbols)]
        for index, ligand in enumerate(excluded_sorted)
    }

    all_ligands_for_plot = sorted(
        set(df['Ligand_Name']) & set(experimental_emax_dict),
        key=ligand_natural_key,
    )
    non_excluded_ligands = [
        ligand for ligand in all_ligands_for_plot if ligand not in excluded_ligands
    ]
    palette = plt.get_cmap('tab20')
    ligand_colors = {
        ligand: palette(index / palette.N)
        for index, ligand in enumerate(non_excluded_ligands)
    }

    for ligand in all_ligands_for_plot:
        ligand_row = df[df['Ligand_Name'] == ligand]
        if ligand_row.empty:
            continue

        ligand_response = pd.to_numeric(
            pd.Series([experimental_emax_dict[ligand]]), errors='coerce'
        ).iloc[0]
        ligand_ddg = pd.to_numeric(ligand_row['DDG'], errors='coerce').iloc[0]
        ligand_ddg_sigma = pd.to_numeric(
            ligand_row['DDG_Error'], errors='coerce'
        ).iloc[0]
        if not all(np.isfinite(value) for value in [ligand_response, ligand_ddg, ligand_ddg_sigma]):
            continue

        is_excluded = ligand in excluded_ligands
        if not is_excluded:
            all_respo.append(ligand_response)
            all_ddg.append(ligand_ddg)
            all_ddg_sigma.append(ligand_ddg_sigma)
        elif not draw_excluded_ligands:
            continue

        if is_excluded:
            marker = excluded_marker_map[ligand]
        elif labels.get(ligand) == 'agonist':
            marker = 'o'
        elif labels.get(ligand) == 'antagonist':
            marker = 's'
        else:
            marker = '^'

        legend_name = ligand.replace('compound-', 'cpd-')
        ligand_class = labels.get(ligand)
        if color_by_class and ligand_class in {'agonist', 'antagonist'}:
            color = 'C1' if ligand_class == 'agonist' else 'C0'
        elif is_excluded:
            color = '0.6'
            legend_name = f'{legend_name}$^*$'
        else:
            color = ligand_colors[ligand]

        print(f'{ligand}: {ligand_ddg:.3f} +/- {ligand_ddg_sigma:.3f}, {ligand_response:.1f}')
        ax.plot(ligand_ddg, ligand_response, marker=marker, linestyle='None', c=color, label=legend_name)
        ax.errorbar(ligand_ddg, ligand_response, xerr=ligand_ddg_sigma, yerr=0, fmt='none', c=color)

    if color_by_class:
        from matplotlib.lines import Line2D
        class_handles = [
            Line2D([0], [0], marker='o', color='C1', linestyle='None', markersize=7, label='agonists'),
            Line2D([0], [0], marker='s', color='C0', linestyle='None', markersize=7, label='antagonists'),
        ]
        if draw_legend:
            ax.legend(handles=class_handles, loc='upper right', fontsize=8)
    elif draw_legend:
        handles, legend_labels = ax.get_legend_handles_labels()
        unique = dict(zip(legend_labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)

    def sigmoid(x, k, x0):
        return 100 / (1 + np.exp(-k * (x - x0)))

    if len(all_ddg) >= 3:
        k, x0 = curve_fit(sigmoid, all_ddg, all_respo, p0=[-1, 1])[0]
        print(f'Sigmoidal fit coefficients: k={k:.3f}, x0={x0:.3f}')
        predictions = sigmoid(np.array(all_ddg), k, x0)
        mse = mean_squared_error(all_respo, predictions)
        r2 = r2_score(all_respo, predictions)
        print('R^2:', r2)
        print('RMSE:', np.sqrt(mse))

        num_samples = 1000
        response_samples = np.array([
            sigmoid(np.random.normal(all_ddg, all_ddg_sigma), k, x0)
            for _ in range(num_samples)
        ])
        expected_rmse = np.sqrt(np.mean(np.std(response_samples, axis=0) ** 2))
        print('Expected RMSE:', expected_rmse)

        xgrid = np.arange(-12, 8, 0.01)
        sfit = sigmoid(xgrid, k, x0)
        ax.plot(xgrid, sfit, ':', color='gray', label='Sigmoidal fit', zorder=1)
        sigma = np.sqrt(mse)
        ax.fill_between(xgrid, sfit - 2 * sigma, sfit + 2 * sigma, color='gray', alpha=0.1, edgecolor=None)
        ax.fill_between(xgrid, sfit - sigma, sfit + sigma, color='gray', alpha=0.1, edgecolor=None, zorder=0)
    else:
        print('Not enough points for sigmoid fit (need at least 3).')

    ax.set_xlim(xlim)
    ax.set_ylim(-10, 110)
    ax.set_ylabel('max. response [%]')
    ax.set_xlabel(
        '$\\Delta G_\\mathrm{A} - \\Delta G_\\mathrm{I}$ [kcal/mol]'
        if difference_label else '$\\Delta \\Delta G$ [kcal/mol]'
    )
    fig.tight_layout()
    if save_as is not None:
        fig.savefig(save_as + '.pdf', dpi=300)
        fig.savefig(save_as + '.png', dpi=300)
    return fig, ax


_,_ = plot_max_response_over_ddg(
    experimental_emax_dict, excluded_ligands,
    color_by_class=False, draw_excluded_ligands=True,
    xlim=[-8.5, 6],
    save_as='../plots/trpml1_max_response_over_ddg',
)

_,_ = plot_max_response_over_ddg(
    experimental_emax_dict, excluded_ligands,
    color_by_class=True, draw_excluded_ligands=False, draw_legend=False, difference_label=True,
    xlim=[-9, 6], figsize=(2.75, 3),
    save_as='../plots/trpml1_max_response_over_ddg_toc',
)


# =================================================
# Experimental vs predicted binding free energy
# =================================================


def plot_experimental_vs_predicted_binding(
    x_column,
    x_label,
    xlim,
    save_as=None,
    add_lower_bound_arrows=False,
    include_excluded_with_missing_x=False,
    ):
    fig, ax = plt.subplots(1, 1, figsize=[4.8, 3.1], dpi=300)

    def ligand_natural_key(lig):
        token = str(lig).split('-', 1)[-1].lower()
        i = 0
        while i < len(token) and token[i].isdigit():
            i += 1
        number = int(token[:i]) if i > 0 else float('inf')
        suffix = token[i:]
        return (number, suffix, token)

    # Use three distinct symbols for excluded ligands (not agonist/antagonist symbols).
    excluded_symbols = ['P', 'X', 'D']
    excluded_sorted = sorted(excluded_ligands, key=ligand_natural_key)
    excluded_marker_map = {lig: excluded_symbols[i % len(excluded_symbols)] for i, lig in enumerate(excluded_sorted)}

    # Convenience: compute the negative p-scale on demand.
    if x_column == 'neg_pPotency' and x_column not in trpml1_electrophysiology.columns:
        exp_potency_uM = pd.to_numeric(trpml1_electrophysiology['potency_uM_min'], errors='coerce')
        exp_potency_M = exp_potency_uM * 1e-6
        trpml1_electrophysiology['neg_pPotency'] = np.log10(exp_potency_M)

    if x_column not in trpml1_electrophysiology.columns:
        raise KeyError(f"Column '{x_column}' not found in trpml1_electrophysiology")

    experimental_x_dict = dict(zip(trpml1_electrophysiology['Compound'], trpml1_electrophysiology[x_column]))

    # Use same ligand ordering and color mapping as in the Emax plot for non-excluded ligands.
    ligands_for_plot = [
        lig
        for lig in sorted(set(df['Ligand_Name']) & set(experimental_x_dict.keys()), key=ligand_natural_key)
        if lig not in excluded_ligands
    ]
    palette = plt.get_cmap('tab20')
    ligand_colors = {lig: palette(i % palette.N) for i, lig in enumerate(ligands_for_plot)}

    x_fit = []
    y_fit = []

    # Place missing excluded points near the right edge as lower-bound markers.
    x_span = xlim[1] - xlim[0]
    missing_excluded_x = xlim[1] - 0.15 * x_span

    for ligand in sorted(set(df['Ligand_Name']) & set(experimental_x_dict.keys()), key=ligand_natural_key):
        ligand_row = df[df['Ligand_Name'] == ligand]
        if ligand_row.empty:
            continue

        exp_x = pd.to_numeric(pd.Series([experimental_x_dict[ligand]]), errors='coerce').iloc[0]
        dga = pd.to_numeric(ligand_row['DGA'], errors='coerce').iloc[0]
        dgi = pd.to_numeric(ligand_row['DGI'], errors='coerce').iloc[0]
        is_excluded = ligand in excluded_ligands

        if not (np.isfinite(dga) and np.isfinite(dgi)):
            continue

        is_missing_excluded = False
        if not np.isfinite(exp_x):
            if is_excluded and include_excluded_with_missing_x:
                exp_x = missing_excluded_x
                is_missing_excluded = True
            else:
                continue

        pred_dg = min(dga, dgi)

        if is_excluded:
            marker = excluded_marker_map[ligand]
        elif labels.get(ligand) == 'agonist':
            marker = 'o'
        elif labels.get(ligand) == 'antagonist':
            marker = 's'
        else:
            marker = '^'

        legend_name = ligand.replace('compound-', 'cpd-')
        if is_excluded:
            color = '0.6'
            legend_name = f"{legend_name} (excl.)"
        else:
            color = ligand_colors[ligand]
            x_fit.append(exp_x)
            y_fit.append(pred_dg)

        ax.scatter(exp_x, pred_dg, c=[color], marker=marker, s=55, edgecolors='none', label=legend_name, zorder=3)
        if is_excluded and (add_lower_bound_arrows or is_missing_excluded):
            # Draw from the point itself to the right edge of the plotting area.
            x_arrow_start = exp_x
            x_arrow_end = xlim[1]
            ax.annotate(
                '',
                xy=(x_arrow_end, pred_dg),
                xytext=(x_arrow_start, pred_dg),
                arrowprops=dict(arrowstyle='-|>', color='0.5', lw=1.2, shrinkA=0, shrinkB=0),
                zorder=1,
            )

    stats_label = 'RMSE: n/a\nR$^2$: n/a'
    if len(x_fit) >= 2:
        slope, intercept = np.polyfit(np.array(x_fit), np.array(y_fit), 1)
        #x_line = np.linspace(xlim[0] - 0.25, np.max(np.array(x_fit)), 200)
        x_line = np.linspace(xlim[0] - 0.25, xlim[1] + 0.25, 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='gray', lw=1.5, linestyle=':', zorder=0)
        # Shade of +/- 1 and +/- 2 kcal/mol around the linear fit line
        ax.fill_between(x_line, y_line - 1, y_line + 1, color='gray', alpha=0.2, linewidth=0, zorder=-1)
        ax.fill_between(x_line, y_line - 2, y_line + 2, color='gray', alpha=0.1, linewidth=0, zorder=-1)
        fit_pred = slope * np.array(x_fit) + intercept
        fit_r2 = r2_score(np.array(y_fit), fit_pred)
        fit_rmse = np.sqrt(mean_squared_error(np.array(y_fit), fit_pred))
        stats_label = f'RMSE: {fit_rmse:.2f} kcal/mol\nR$^2$: {fit_r2:.2f}'
        print(f"{x_column}: Linear fit R^2: {fit_r2:.3f}, RMSE: {fit_rmse:.3f} kcal/mol")
    else:
        print(f"{x_column}: Not enough non-excluded points for linear fit (need at least 2).")

    ax.set_xlim(xlim)
    ax.set_ylim(-23, -9)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Predicted $\\Delta G$ [kcal/mol]')
    ax.text(
        0.05, 0.94, stats_label,
        transform=ax.transAxes, ha='left', va='top', fontsize=10,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.25')
    )

    handles, legend_labels = ax.get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)

    fig.tight_layout()
    if save_as is not None:
        fig.savefig(save_as + '.pdf', dpi=300)
        fig.savefig(save_as + '.png', dpi=300)


# Approximate model to get actual binding free energies (kcal/mol) out of EC50 and IC50 values
# Approximate Ki as IC50 
# Agonists: Ki = EC50 / (1 - Emax/100%).

def calculate_approx_binding_dG(row):
    if row['potency_source'] == 'agonist_ec50':
        ec50_uM = pd.to_numeric(pd.Series([row['agonist_ec50_uM']]), errors='coerce').iloc[0]
        emax_value = pd.to_numeric(pd.Series([row['emax']]), errors='coerce').iloc[0]
        if not (np.isfinite(ec50_uM) and np.isfinite(emax_value)):
            return np.nan

        ec50_m = ec50_uM * 1e-6
        emax_fraction = emax_value / 100.0
        if emax_fraction >= 1.0:
            return np.nan  # Avoid division by zero or negative values
        ki = ec50_m / (1 - emax_fraction)
    elif row['potency_source'] == 'antagonist_ic50':
        ic50_uM = pd.to_numeric(pd.Series([row['antagonist_ic50_uM']]), errors='coerce').iloc[0]
        if not np.isfinite(ic50_uM):
            return np.nan

        ic50_m = ic50_uM * 1e-6
        ki = ic50_m / 1.0  # Approximation for Ki
    else:
        return np.nan

    if ki <= 0 or not np.isfinite(ki):
        return np.nan  # Avoid log of non-positive or invalid values

    dG_kcal_mol = R_KCAL_PER_MOL_K * TEMPERATURE_K * np.log(ki)
    return dG_kcal_mol

trpml1_electrophysiology['approx_binding_dG_kcal_mol'] = trpml1_electrophysiology.apply(
    lambda row: calculate_approx_binding_dG(row),
    axis=1
)
display(trpml1_electrophysiology[['Compound', 'potency_source', 'emax', 'approx_binding_dG_kcal_mol']])


plot_experimental_vs_predicted_binding(
    x_column='neg_pPotency',
    x_label='Exp. potency $-$pIC$_{50}$, $-$pEC$_{50}$',
    xlim=(-7.75, -4.25),
    save_as='../plots/trpml1_binding_neg_p-potency',
    add_lower_bound_arrows=True,
    include_excluded_with_missing_x=False,
 )

plot_experimental_vs_predicted_binding(
    x_column='approx_binding_dG_kcal_mol',
    x_label='Approx. experimental binding $\\Delta G$ [kcal/mol]',
    xlim=(-9.5, -6.0),
    save_as='../plots/trpml1_binding_approx_dg',
    add_lower_bound_arrows=True,
    include_excluded_with_missing_x=True,
 )
