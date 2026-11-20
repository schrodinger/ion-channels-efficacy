import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score


# Used in notebooks elsewhere in the project.
def ddg_barplot(data, labels, figsize=(4, 3), errorbars=False, constant_errorbar=None, ylim=None, file_name=None, xlabel_mode='dataframe', ligand_names=None):
    """
    Create a bar plot of ΔΔG values for ligands with specified labels.

    Parameters:
    - data: DataFrame containing ligand names, ΔΔG values, and errors (column names: 'Ligand_Name', 'DDG', 'DDG_Error').
      The 'Ligand_Name' column is required and must use the capitalized name exactly.
    - labels: Dictionary mapping ligand names to labels ('agonist' or 'antagonist').
    - figsize: Tuple specifying the figure size.
    - errorbars: Boolean indicating whether to display error bars.
    - constant_errorbar: Optional constant value for error bars.
    - ylim: Optional tuple specifying y-axis limits.
    - file_name: Optional file name to save the figure.
    - xlabel_mode: 'numbers' (ligand index), 'dataframe' (ligand_name from data), or 'dict' (values from ligand_names dictionary).
    - ligand_names: Optional dictionary mapping ligand names to display names.

    Returns:
    - fig, ax: Matplotlib figure and axis objects.
    """

    # barplot of ddg values with labels, only for ligands with label agonist or antagonist
    bar_ddg = data[
        data['Ligand_Name'].isin([
            lig for lig, label in labels.items() if label in ['agonist', 'antagonist']
        ])
    ]

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    for i, row in bar_ddg.iterrows():
        ligand = row['Ligand_Name']
        ddg = row['DDG']
        ddg_error = row['DDG_Error']
        if labels[ligand] == 'agonist':
            color = 'C1'
        else:
            color = 'C0'

        if not errorbars:
            yerr = None
        elif constant_errorbar is not None:
            yerr = float(constant_errorbar)
        else:
            yerr = ddg_error

        bar_alpha = 0.5 if errorbars else 1.0
        ax.bar(ligand, ddg, color=color, alpha=bar_alpha, yerr=yerr, ecolor=color, capsize=3)

    ax.set_xticks(range(len(bar_ddg['Ligand_Name'])))
    if xlabel_mode == 'numbers':
        ax.set_xticklabels(1 + np.arange(len(bar_ddg['Ligand_Name']), dtype=int))
        ax.set_xlabel('Ligand #', size=12)
    elif xlabel_mode == 'none':
        ax.set_xticklabels([])
        ax.set_xlabel('Ligand', size=12)
    elif xlabel_mode == 'dataframe':
        ax.set_xticklabels(bar_ddg['Ligand_Name'], rotation=90)
    elif xlabel_mode == 'dict' and ligand_names is not None:
        ax.set_xticklabels([ligand_names[lig] for lig in bar_ddg['Ligand_Name']], rotation=90)
    else:
        print("Invalid xlabel_mode or missing ligand_names dictionary. Defaulting to dataframe labels.")
        ax.set_xticklabels(bar_ddg['Ligand_Name'], rotation=90)

    ax.set_ylabel(r'$\Delta\Delta G$ [kcal/mol]', size=12)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if file_name:
        fig.savefig(f'{file_name}.png', dpi=300, format='png')
        fig.savefig(f'{file_name}.pdf', dpi=300, format='pdf')
    return fig, ax


def combine_dataframes(df_a, df_i):
    """Combine active-state and inactive-state data by ligand while keeping the minimum-energy values."""
    df_a = df_a.copy()
    df_i = df_i.copy()

    ligands = list(np.unique(df_a['ligand_name']))
    df = pd.DataFrame(columns=['Ligand_Name', 'DDG', 'DGA', 'DGI', 'DDG_Error', 'DGA_Error', 'DGI_Error'])

    for ligand in ligands:
        dg_a = pd.to_numeric(
            df_a.loc[df_a['ligand_name'] == ligand, 'pred_dg'],
            errors='coerce',
        ).dropna()
        dg_a_sigma = pd.to_numeric(
            df_a.loc[df_a['ligand_name'] == ligand, 'pred_dg_error'],
            errors='coerce',
        ).dropna() if 'pred_dg_error' in df_a.columns else pd.Series(index=dg_a.index, dtype=float)
        dg_i = pd.to_numeric(
            df_i.loc[df_i['ligand_name'] == ligand, 'pred_dg'],
            errors='coerce',
        ).dropna()
        dg_i_sigma = pd.to_numeric(
            df_i.loc[df_i['ligand_name'] == ligand, 'pred_dg_error'],
            errors='coerce',
        ).dropna() if 'pred_dg_error' in df_i.columns else pd.Series(index=dg_i.index, dtype=float)

        try:
            dga_idx = dg_a.idxmin()
            dga = dg_a.loc[dga_idx]
            dga_error = dg_a_sigma.loc[dga_idx] if dga_idx in dg_a_sigma.index else np.nan
        except (ValueError, TypeError, IndexError):
            dga = 'failed'
            dga_error = np.nan

        try:
            dgi_idx = dg_i.idxmin()
            dgi = dg_i.loc[dgi_idx]
            dgi_error = dg_i_sigma.loc[dgi_idx] if dgi_idx in dg_i_sigma.index else np.nan
        except (ValueError, TypeError, IndexError):
            dgi = 'failed'
            dgi_error = np.nan

        if isinstance(dga, str) or isinstance(dgi, str):
            ddg = np.nan
            ddg_error = np.nan
        else:
            ddg = dga - dgi
            ddg_error = np.sqrt(dga_error**2 + dgi_error**2)

        df = pd.concat([
            df,
            pd.DataFrame({
                'Ligand_Name': [ligand],
                'DDG': [ddg],
                'DGA': [dga],
                'DGI': [dgi],
                'DDG_Error': [ddg_error],
                'DGA_Error': [dga_error],
                'DGI_Error': [dgi_error],
            }),
        ], ignore_index=True)

    return df


# ======================================================
# Sigmoidal max. response model: fit and leave-one-out
# ======================================================
#
# Model: Emax = a / (1 + exp(-k * (DDG - x0))). The plateau `a` is not fitted. It is either fixed to a
# given number (e.g. 100) or, with plateau='max', set to the largest measured Emax of the ligands
# used in the fit. The `data` DataFrames have the columns Ligand_Name, DDG, DDG_Error and Emax.


def sigmoid(x, k, x0, a):
    return a / (1 + np.exp(-k * (x - x0)))


def fit_sigmoid(ddg, emax, plateau, p0=(-1, 1)):
    """Least-squares fit of the two-parameter sigmoid (k, x0) with a fixed plateau.

    `plateau` is a number or 'max' (largest value in `emax`). With 'max', a ligand that exceeds all
    others is an overshoot of the model's ceiling. Returns (k, x0), their covariance matrix and the
    plateau that was used.
    """
    if isinstance(plateau, str):
        if plateau != 'max':
            raise ValueError("plateau must be a number or 'max'")
        plateau = np.max(emax)
    params, pcov = curve_fit(lambda x, k, x0: sigmoid(x, k, x0, plateau), ddg, emax, p0=p0)
    return params, pcov, plateau


def fit_emax_model(data, plateau, num_samples=1000, seed=0):
    """Fit the sigmoid to all ligands in `data` and collect summary statistics."""
    ddg, emax, ddg_sigma = data['DDG'].to_numpy(), data['Emax'].to_numpy(), data['DDG_Error'].to_numpy()
    params, pcov, plateau = fit_sigmoid(ddg, emax, plateau)
    param_errors = np.sqrt(np.diag(pcov))
    predictions = sigmoid(ddg, *params, plateau)
    rng = np.random.default_rng(seed)
    response_samples = np.array([
        sigmoid(rng.normal(ddg, ddg_sigma), *params, plateau) for _ in range(num_samples)
    ])
    return {
        'n': len(data),
        'k': params[0], 'k_err': param_errors[0],
        'x0': params[1], 'x0_err': param_errors[1],
        'a': plateau,
        'r2': r2_score(emax, predictions),
        'rmse': np.sqrt(mean_squared_error(emax, predictions)),
        # RMSE expected from the DDG uncertainty alone
        'expected_rmse': np.sqrt(np.mean(np.std(response_samples, axis=0) ** 2)),
    }


def leave_one_out_emax(data, plateau):
    """Refit the sigmoid n times, each time predicting the held-out ligand.

    With plateau='max' the ceiling is recomputed from the training ligands in every round, so a
    held-out ligand that exceeds all others is predicted with a lower ceiling.
    """
    rows = []
    for i, ligand in enumerate(data['Ligand_Name']):
        train = data.drop(index=i)
        try:
            (k, x0), _, train_plateau = fit_sigmoid(train['DDG'], train['Emax'], plateau)
        except RuntimeError:  # fit did not converge
            k, x0, train_plateau = np.nan, np.nan, np.nan
        rows.append({
            'Ligand_Name': ligand,
            'DDG': data.at[i, 'DDG'],
            'Emax': data.at[i, 'Emax'],
            'Emax_pred_loo': sigmoid(data.at[i, 'DDG'], k, x0, train_plateau),
            'k': k, 'x0': x0, 'a': train_plateau,
        })
    result = pd.DataFrame(rows)
    result['error'] = result['Emax_pred_loo'] - result['Emax']
    return result


def summarize_loo(loo):
    valid = loo.dropna(subset=['Emax_pred_loo'])
    return {
        'n': len(valid),
        'n_failed': len(loo) - len(valid),
        'rmse': np.sqrt(np.mean(valid['error'] ** 2)),
        'mae': np.mean(np.abs(valid['error'])),
        # Q^2: 1 - PRESS / total sum of squares (analogue of R^2 for held-out predictions)
        'q2': r2_score(valid['Emax'], valid['Emax_pred_loo']),
    }
