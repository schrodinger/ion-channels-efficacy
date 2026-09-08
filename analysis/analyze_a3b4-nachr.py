# Generated from Analysis-a3b4-nAChR.ipynb.
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
from sklearn.metrics import mean_squared_error, r2_score
from analysis_utils import combine_dataframes, ddg_barplot


# =================
# Overview Plot
# =================


# Load labels and alternative names
labels_data = pd.read_csv('../labels/a3b4-nachr_labels.csv')
names = labels_data.set_index('ligand')['name'].to_dict()
labels = labels_data.set_index('ligand')['label'].to_dict()

# Load AB-FEP results
df_a = pd.read_csv('../results/a3b4-nachr-a.csv')
df_i = pd.read_csv('../results/a3b4-nachr-i.csv')

# Merge AB-FEP results and sort by DDG
merged_df = combine_dataframes(df_a, df_i)
merged_df = merged_df.sort_values(by='DDG', ascending=True)
display(merged_df)


_,_ = ddg_barplot(
    merged_df, labels, figsize=(4, 3), errorbars=False, xlabel_mode='none', ylim=(-3, 7),
    file_name='../plots/a3b4-nachr_barplot_nolabels.png'
)
_,_ = ddg_barplot(
    merged_df, labels, figsize=(4, 3), errorbars=True, xlabel_mode='none', ylim=(-3, 7),
    file_name='../plots/a3b4-nachr_barplot_nolabels_bennett-error.png'
)
_,_ = ddg_barplot(
    merged_df, labels, figsize=(4, 3), errorbars=True, xlabel_mode='none', constant_errorbar=np.sqrt(2), ylim=(-3, 7),
    file_name='../plots/a3b4-nachr_barplot_nolabels_constant-error.png'
)
_,_ = ddg_barplot(
    merged_df, labels, figsize=(4, 3.5), errorbars=False, xlabel_mode='dict', ligand_names=names, ylim=(-3, 7),
    file_name='../plots/a3b4-nachr_barplot_ligand-names.png'
)
_,_ = ddg_barplot(
    merged_df, labels, figsize=(4, 3.5), errorbars=True, xlabel_mode='dict', ligand_names=names, ylim=(-4.5, 8.5),
    file_name='../plots/a3b4-nachr_barplot_ligand-names_bennett-error.png'
)
_,_ = ddg_barplot(
    merged_df, labels, figsize=(4, 3.5), errorbars=True, xlabel_mode='dict', ligand_names=names, ylim=(-4.5, 8.5), constant_errorbar=np.sqrt(2),
    file_name='../plots/a3b4-nachr_barplot_ligand-names_constant-error.png'
)


# ========================
# Uncertainty Analysis
# ========================


from scipy.stats import norm

# Calculate binary RMSE and R2 score for agonists vs antagonists
# using P(agonist) = integral of N(DDG, DDG_error^2) from -inf to 0.
def ddg_to_agonist_prob(ddg, ddg_error):
    ddg = np.asarray(ddg, dtype=float)
    ddg_error = np.asarray(ddg_error, dtype=float)

    probs = np.empty_like(ddg, dtype=float)
    positive_sigma = ddg_error > 0

    # Probabilistic case: integrate Gaussian CDF up to 0.
    #probs[positive_sigma] = norm.cdf(0.0, loc=ddg[positive_sigma], scale=ddg_error[positive_sigma])
    probs[positive_sigma] = norm.cdf(-ddg[positive_sigma], scale=ddg_error[positive_sigma])

    # Deterministic fallback when uncertainty is zero or invalid.
    probs[~positive_sigma] = (ddg[~positive_sigma] < 0).astype(float)
    probs[~positive_sigma & (ddg == 0)] = 0.5

    return probs

# q_i = Phi(y_i * d_i / sigma_i): probability ligand i keeps its correct sign.
def probability_of_correct_sign(d, sigma, y):
    d = np.asarray(d, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    y = np.asarray(y, dtype=float)
    q = np.empty_like(d, dtype=float)
    positive_sigma = sigma > 0

#    q[positive_sigma] = norm.cdf((y[positive_sigma] * d[positive_sigma]) / sigma[positive_sigma])
#    q[positive_sigma] = norm.cdf((y[positive_sigma] * d[positive_sigma]) / (np.sqrt(2)*sigma[positive_sigma]))
    cdf_sigma = norm.cdf((y[positive_sigma] * d[positive_sigma]) / sigma[positive_sigma])
    cdf_sqrt2 = norm.cdf((y[positive_sigma] * d[positive_sigma]) / (sigma[positive_sigma] * np.sqrt(2)))
    q[positive_sigma] = 0.5*(1 + cdf_sqrt2**2/cdf_sigma)

    # Deterministic fallback when uncertainty is zero or invalid.
    nonpositive_sigma = ~positive_sigma
    signed_d = y[nonpositive_sigma] * d[nonpositive_sigma]
    q[nonpositive_sigma] = (signed_d > 0).astype(float)
    q[nonpositive_sigma & ((y * d) == 0)] = 0.5

    return q

def probability_of_max_n_wrong_signs(d, sigma, y_sign, n=1):
    """Calculate probability for at most n wrong signs in a set of ligands with given DDG values and uncertainties."""
    q_i = probability_of_correct_sign(d, sigma, y_sign)  # P(correct) for each ligand
    N = len(q_i)
    dp = np.zeros(N+1, dtype=float)
    dp[0] = 1.0

    for i in range(N):
        f_i = 1.0 - q_i[i]  # P(wrong)
        for j in range(min(i+1, n), -1, -1):
            dp[j] = dp[j] * q_i[i] + (dp[j-1] * f_i if j > 0 else 0.0)
    # Sum up to n wrong signs to get the probability of at most n wrong signs.
    dp[:n+1] = np.cumsum(dp[:n+1])
    return dp[:n+1]



agonists = merged_df[merged_df['Ligand_Name'].isin([lig for lig, label in labels.items() if label == 'agonist'])]
antagonists = merged_df[merged_df['Ligand_Name'].isin([lig for lig, label in labels.items() if label == 'antagonist'])]

y_true = np.concatenate([np.ones(len(agonists)), np.zeros(len(antagonists))])

y_pred = np.concatenate([
    ddg_to_agonist_prob(agonists['DDG'].values, agonists['DDG_Error'].values),
    ddg_to_agonist_prob(antagonists['DDG'].values, antagonists['DDG_Error'].values)
])

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

# Build y_i in {-1, +1}: agonist expects DDG < 0, antagonist expects DDG > 0.
y_sign = np.concatenate([
    -np.ones(len(agonists)),
    np.ones(len(antagonists))
])
d_all = np.concatenate([
    agonists['DDG'].values,
    antagonists['DDG'].values
])
sigma_all = np.concatenate([
    agonists['DDG_Error'].values,
    antagonists['DDG_Error'].values
])

q_i = probability_of_correct_sign(d_all, sigma_all, y_sign)
#print(f"q_i: {q_i}")
p_all_correct = np.prod(q_i)

q_i_0325kcalmol = probability_of_correct_sign(d_all, np.sqrt(2)*0.325, y_sign)
p_all_correct_0325kcalmol = np.prod(q_i_0325kcalmol)

q_i_1kcalmol = probability_of_correct_sign(d_all, np.sqrt(2)*1.0, y_sign)
#print(f"q_i_1kcalmol: {q_i_1kcalmol}")
p_all_correct_1kcalmol = np.prod(q_i_1kcalmol)

print(f"Binary RMSE (probability-based): {rmse:.4f}")
print(f"Binary R2 score (probability-based): {r2:.4f}")
print('')

print(f"P(all {len(q_i)} classifications correct | given uncertainties): {p_all_correct:.2f}")
print(f"P(all {len(q_i_0325kcalmol)} classifications correct | 0.325 kcal/mol DG uncertainty): {p_all_correct_0325kcalmol:.2f}")
print(f"P(all {len(q_i_1kcalmol)} classifications correct | 1 kcal/mol DG uncertainty): {p_all_correct_1kcalmol:.2f}")
print('')

p_1_wrong = probability_of_max_n_wrong_signs(d_all, sigma_all, y_sign, n=1)
p_1_wrong_0325kcalmol = probability_of_max_n_wrong_signs(d_all, sigma_all*0+np.sqrt(2)*0.325, y_sign, n=1)
p_1_wrong_1kcalmol = probability_of_max_n_wrong_signs(d_all, sigma_all*0+np.sqrt(2)*1.0, y_sign, n=1)
print(f"P(max 1 wrong classification | given uncertainties): {p_1_wrong[1]:.2f}")
print(f"P(max 1 wrong classification | 0.325 kcal/mol DG uncertainty): {p_1_wrong_0325kcalmol[1]:.2f}")
print(f"P(max 1 wrong classification | 1 kcal/mol DG uncertainty): {p_1_wrong_1kcalmol[1]:.2f}")


# Correlated error model for DDG = G_a - G_i.
def ddg_error_with_rho(err_a, err_i, rho, stat_a=None, stat_i=None):
    err_a = np.asarray(err_a, dtype=float)
    err_i = np.asarray(err_i, dtype=float)
    rho = np.asarray(rho, dtype=float)

    # Default: no additional statistical component.
    if stat_a is None:
        stat_a = np.zeros_like(err_a, dtype=float)
    else:
        stat_a = np.asarray(stat_a, dtype=float)

    if stat_i is None:
        stat_i = np.zeros_like(err_i, dtype=float)
    else:
        stat_i = np.asarray(stat_i, dtype=float)

    # Var(G_a - G_i) = err_a^2 + err_i^2 - 2*rho*err_a*err_i + stat_a^2 + stat_i^2
    var_ddg = err_a**2 + err_i**2 - 2.0 * rho * err_a * err_i + stat_a**2 + stat_i**2
    var_ddg = np.maximum(var_ddg, 0.0)
    return np.sqrt(var_ddg)


def p_all_correct_for_rho(rho_values, d_all, err_a, err_i, y_sign, stat_a=None, stat_i=None):
    rho_values = np.asarray(rho_values, dtype=float)
    p_all = np.empty_like(rho_values, dtype=float)

    for k, rho in enumerate(rho_values):
        sigma_rho = ddg_error_with_rho(err_a, err_i, rho, stat_a=stat_a, stat_i=stat_i)
        q_i_rho = probability_of_correct_sign(d_all, sigma_rho, y_sign)
        p_all[k] = np.prod(q_i_rho)

    return p_all


def p_at_most_n_wrong_for_rho(rho_values, d_all, err_a, err_i, y_sign, n_wrong, stat_a=None, stat_i=None):
    """Return P(number of wrong-sign predictions <= n_wrong) for each rho.

    Uses the Poisson-binomial exact sum via dynamic programming:
    dp[k] = probability of exactly k wrong predictions.
    """
    rho_values = np.asarray(rho_values, dtype=float)
    p_atmost = np.empty_like(rho_values, dtype=float)
    N = len(d_all)

    for k, rho in enumerate(rho_values):
        sigma_rho = ddg_error_with_rho(err_a, err_i, rho, stat_a=stat_a, stat_i=stat_i)
        q_i = probability_of_correct_sign(d_all, sigma_rho, y_sign)  # P(correct) for each ligand
        f_i = 1.0 - q_i                                      # P(wrong)

        # dp[j] = P(exactly j wrong) after processing i ligands.
        max_track = min(n_wrong + 1, N + 1)
        dp = np.zeros(max_track + 1)
        dp[0] = 1.0
        for q, f in zip(q_i, f_i):
            # Iterate in reverse to avoid using updated values.
            for j in range(min(max_track, N), 0, -1):
                dp[j] = dp[j] * q + dp[j - 1] * f
            dp[0] *= q

        p_atmost[k] = dp[:n_wrong + 1].sum()

    return p_atmost


def rho_at_target_probability(rho_values, p_values, target):
    order = np.argsort(p_values)
    p_sorted = p_values[order]
    rho_sorted = rho_values[order]

    p_min = float(p_sorted[0])
    p_max = float(p_sorted[-1])
    if not (p_min <= target <= p_max):
        return np.nan

    return float(np.interp(target, p_sorted, rho_sorted))


def calculate_p_rho_curves(rho_grid, d_all, err_a_all, err_i_all, y_sign, stat_a_all, stat_i_all, thresholds=[0.5]):
    
    p_all_grid = p_all_correct_for_rho(
        rho_grid, d_all, err_a_all, err_i_all, y_sign,
        stat_a=stat_a_all, stat_i=stat_i_all
    )

    p_atmost_1_grid = p_at_most_n_wrong_for_rho(
        rho_grid, d_all, err_a_all, err_i_all, y_sign, n_wrong=1,
        stat_a=stat_a_all, stat_i=stat_i_all
    )

    p_atmost_2_grid = p_at_most_n_wrong_for_rho(
        rho_grid, d_all, err_a_all, err_i_all, y_sign, n_wrong=2,
        stat_a=stat_a_all, stat_i=stat_i_all
    )

    rho_thresholds = {t: rho_at_target_probability(rho_grid, p_all_grid, t) for t in thresholds}

    #print(f"Assumed per-DG uncertainty: {fixed_dg_error:.1f} kcal/mol")
    print(f"rho sweep range: [{rho_grid[0]:.3f}, {rho_grid[-1]:.3f}] with {len(rho_grid)} points")
    print(f"P(all correct) range over sweep: [{p_all_grid.min():.6f}, {p_all_grid.max():.6f}]")
    for t in thresholds:
        rho_t = rho_thresholds[t]
        if np.isnan(rho_t):
            print(f"target P = {t:.4f}: not reachable in rho in [-0.999, 0.999]")
        else:
            print(f"target P = {t:.4f}: rho ≈ {rho_t:.4f}")

    p_grids = {
        'P(all correct | ρ)':  p_all_grid,
        'P(≤ 1 wrong | ρ)':    p_atmost_1_grid,
        'P(≤ 2 wrong | ρ)':    p_atmost_2_grid,
    }

    return p_grids


# Rho Grid
rho_grid = np.linspace(-0.9999, 0.9999, 20001)

# Fixed per-DG baseline uncertainty (e.g., model/systematic component).
fixed_dg_error = 1.0
err_a_all = np.full_like(d_all, fixed_dg_error, dtype=float)
err_i_all = np.full_like(d_all, fixed_dg_error, dtype=float)

# Add sampling errors from dataframe as statistical components.
stat_a_all = np.concatenate([
    agonists['DGA_Error'].values,
    antagonists['DGA_Error'].values
]).astype(float)
stat_i_all = np.concatenate([
    agonists['DGI_Error'].values,
    antagonists['DGI_Error'].values
]).astype(float)

p_rho_bennett = calculate_p_rho_curves(rho_grid, d_all, err_a_all, err_i_all, y_sign, stat_a_all, stat_i_all)


# Fixed per-DG statistical uncertainty
sigma_stat = 0.46/np.sqrt(2)
print(f'Uniform statistical uncertainty:{sigma_stat:.3f}')
sigma_stat_i = np.zeros(len(stat_i_all))+sigma_stat
sigma_stat_a = np.zeros(len(stat_a_all))+sigma_stat

# The systematic error needs to be adapted to account for the added statistical uncertainty, so that the total per-DG uncertainty remains 1 kcal/mol.
fixed_dg_error_mod = np.sqrt(1.0 - sigma_stat**2)
print(f'Adapted systematic uncertainty: {fixed_dg_error_mod:.3f}')
err_a_all_mod = np.full_like(d_all, fixed_dg_error_mod, dtype=float)
err_i_all_mod = np.full_like(d_all, fixed_dg_error_mod, dtype=float)

p_rho_sampled = calculate_p_rho_curves(rho_grid, d_all, err_a_all_mod, err_i_all_mod, y_sign, sigma_stat_i, sigma_stat_a, thresholds=[0.5])


def plot_p_vs_rho(rho_grid, p_grids, thresholds=None, max_lines=3,
                  xlim=(0.0, 1.0), ylim=(0.0, 1.0),
                  figsize=(7.75, 3.75), dpi=300,
                  save_as=None):
    """Plot one or more P(<=n wrong | rho) curves against the rho grid.
    """
    threshold_colors = {
        0.5:    '#B22222',
        0.6827: '#1F77B4',
        0.9545: '#2E8B57',
    }

    curve_styles = [
        dict(color='black',  linewidth=2.5, linestyle='-'),
        dict(color='gray',   linewidth=2.5, linestyle='--'),
        dict(color='gray',   linewidth=2.5, linestyle=':'),
        dict(color='silver', linewidth=2.0, linestyle='-.'),
    ]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for idx, (label, p_grid) in enumerate(p_grids.items()):
        if idx < max_lines:
            style = curve_styles[idx % len(curve_styles)]
            ax.plot(rho_grid, p_grid, label=label, **style)

    if thresholds:
        # Default: derive rho crossings from the first p_grid.
        first_p = next(iter(p_grids.values()))
        rho_thresholds = {
            t: rho_at_target_probability(rho_grid, first_p, t)
            for t in thresholds
        }
        print('Rho thresholds:', rho_thresholds)

        for target in thresholds:
            color = threshold_colors.get(target, 'gray')
            ax.axhline(target, color=color, linestyle='--', linewidth=1.3, alpha=0.85)

            rho_target = rho_thresholds.get(target, np.nan)
            if np.isfinite(rho_target):
                ax.axvline(rho_target, color=color, linestyle=':', linewidth=1.3, alpha=0.9)
                ax.scatter([rho_target], [target], color=color, s=35, zorder=4)
 #               ax.annotate(f"P={target:.2f} at ρ={rho_target:.2f}", xy=(0.6, target), xytext=(7, -7), ha='center', va='top', textcoords='offset points', fontsize=12, color=color)

    ax.set_xlabel('Correlation coefficient ρ', size=12)
    ax.set_ylabel('P(≤ n wrong)', size=12)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc='upper left', frameon=True, framealpha=1, fontsize=12)
    plt.tight_layout()
    if save_as:
        fig.savefig(save_as+'.png', dpi=dpi)
        fig.savefig(save_as+'.pdf', dpi=dpi)
    return fig, ax


# Plot P(all classifications correct) as a function of rho.
fig, ax = plot_p_vs_rho(
    rho_grid,
    p_rho_bennett,
    thresholds=[0.5],
    max_lines=2,
    save_as='../plots/a3b4-nachr_uncertainty_p-vs-rho_bennett'
)


# Plot P(all classifications correct) as a function of rho.
fig, ax = plot_p_vs_rho(
    rho_grid,
    p_rho_sampled,
    thresholds=[0.5],
    max_lines=2,
    save_as='../plots/a3b4-nachr_uncertainty_p-vs-rho_sampled'
)


"""Plot a Gaussian centered at ΔΔG = -2 with σ = 1."""

MU = -2.0
SIGMA = 1.0


def gaussian(x: np.ndarray) -> np.ndarray:
    """Return the normalized Gaussian density at x."""
    return np.exp(-0.5 * ((x - MU) / SIGMA) ** 2) / (
        SIGMA * np.sqrt(2.0 * np.pi)
    )

x = np.linspace(MU - 5.0 * SIGMA, MU + 5.0 * SIGMA, 2_000)
y = gaussian(x)
peak = float(gaussian(np.array([MU]))[0])

fig, ax = plt.subplots(figsize=(8.0, 4.8))
mask = x <= 0.0
ax.fill_between(x[mask], y[mask], 0.0, color="#808080", alpha=0.40, linewidth=0)
ax.plot(x, y, color="#000000", linewidth=5.0)
ax.vlines(MU, 0.0, peak, color="#9A3E43", linewidth=5.0)
ax.annotate(r"$\Delta\Delta G$", xy=(MU, 0.0), xytext=(0, -12), textcoords="offset points", ha="center", va="top", fontsize=36, color="#9A3E43")
sigma_height = float(gaussian(np.array([MU + SIGMA]))[0])
ax.hlines(sigma_height, MU, MU + SIGMA, color="#2F6B3C", linewidth=5.0)
ax.annotate(r"$\sigma$", xy=(MU + SIGMA / 2.0, sigma_height), xytext=(0, -8), textcoords="offset points", ha="center", va="top", fontsize=36, color="#2F6B3C")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_position(("data", 0.0))
ax.spines["left"].set_position(("data", 0.0))
ax.spines["bottom"].set_linewidth(2.4)
ax.spines["left"].set_linewidth(2.4)
ax.set_xticks([])
ax.set_yticks([])
ax.annotate("0", xy=(0.0, 0.0), xytext=(0, -12), textcoords="offset points", ha="center", va="top", fontsize=36)
ax.set_xlim(MU - 4.0 * SIGMA, MU + 4.0 * SIGMA)
ax.set_ylim(0.0, 1.2 * peak)
fig.tight_layout()
fig.savefig('../plots/uncertainty_gaussian_plot.png', dpi=300)
fig.savefig('../plots/uncertainty_gaussian_plot.pdf', dpi=300)
