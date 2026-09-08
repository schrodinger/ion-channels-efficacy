# Generated from Analysis-HTR3A.ipynb.
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
from analysis_utils import combine_dataframes


# -------------
# Load Data
# -------------


electrophysiology = pd.read_csv('../labels/htr3a_electrophysiology.csv')
ki = electrophysiology.set_index('ligand')['ki_uM'].to_dict()
response = electrophysiology.set_index('ligand')['response_pct'].astype(float).to_dict()


# 8FSB is the active state and 6W1Y is the inactive state.
active_data = pd.read_csv('../results/htr3a-8fsb.csv')
inactive_data = pd.read_csv('../results/htr3a-6w1y.csv')
data_htr3a = combine_dataframes(active_data, inactive_data).sort_values('DDG').reset_index(drop=True)
display(data_htr3a)


# ====================
# Maximum Response
# ====================


# Prepare the data for plotting
plot_data = data_htr3a.set_index('Ligand_Name').loc[response.keys()]
all_respo = np.array(list(response.values()))
all_ddg = plot_data['DDG'].to_numpy()
all_ddg_sigma = plot_data['DDG_Error'].to_numpy()

# Create the figure and axis for the plot
fig, ax = plt.subplots(1,1,figsize=[5,3.1], dpi=300) # [5,3.1]

# Plot each data point with error bars
for i, (lig, ddg, sigma, respo) in enumerate(zip(response.keys(), all_ddg, all_ddg_sigma, all_respo)):
    ax.plot(ddg, respo, 'o', c=f'C{i}', label=lig)
    ax.errorbar(ddg, respo, xerr=sigma, yerr=0, marker=None, c=f'C{i}')

def sigmoid(x, k, x0, a):
    return a / (1 + np.exp(-k * (x - x0)))
k, x0, a = curve_fit(sigmoid, all_ddg, all_respo, p0=[-1, 1, 120])[0]
print('Sigmoidal fit coefficients:', k, x0, a)

predictions = sigmoid(all_ddg, k, x0, a)
mse = mean_squared_error(all_respo, predictions)
r2 = r2_score(all_respo, predictions)
print('R^2:', r2)
print('RMSE:', np.sqrt(mse))

num_samples = 1000
for n in range(num_samples):
    sample_ddg = np.random.normal(all_ddg, all_ddg_sigma)
    sample_response = sigmoid(sample_ddg, k, x0, a)
    if n == 0:
        response_samples = sample_response
    else:
        response_samples = np.vstack((response_samples, sample_response))
expected_rmse = np.sqrt(np.mean(np.std(response_samples, axis=0)**2))
print('Expected RMSE:', expected_rmse)

xgrid = np.arange(-7,7,0.01)
sfit = sigmoid(xgrid, k, x0, a)
ax.plot(xgrid, sfit, ':', color='gray', label='Sigmoidal fit', zorder=1)
sigma = np.sqrt(mse)
ax.fill_between(xgrid, sfit-2*sigma, sfit+2*sigma, color='gray', alpha=0.1, edgecolor=None)
ax.fill_between(xgrid, sfit-1*sigma, sfit+1*sigma, color='gray', alpha=0.1, edgecolor=None, zorder=0)

ax.set_ylim(-10,130)
ax.set_xlim(-5.5,5.5)
ax.set_ylabel('max. response [%]')
ax.set_xlabel('$\\Delta \\Delta G$ [kcal/mol]')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()

saveas ='../plots/htr3a_maximum-response'
fig.savefig(saveas + '.pdf', dpi=300)
fig.savefig(saveas + '.png', dpi=300)


# ====================
# Binding Affinity
# ====================


def plot_dg_comparison(df_for_plot, ki, column="MIN(DG)", save_as=None, dpi=300):
    R = 0.001987
    T = 298.15
    dg_from_ki = {ligand: R * T * np.log(ki_value * 1e-9) for ligand, ki_value in ki.items()}
    ligands = list(dg_from_ki.keys())
    dg_exp = [dg_from_ki[ligand] for ligand in ligands]

    if column == "all":
        dg_fep = [df_for_plot.loc[df_for_plot['Ligand_Name'] == ligand, 'MIN(DG)'].values[0] for ligand in ligands]
        dg_fep_a = [df_for_plot.loc[df_for_plot['Ligand_Name'] == ligand, 'DGA'].values[0] for ligand in ligands]
        dg_fep_b = [df_for_plot.loc[df_for_plot['Ligand_Name'] == ligand, 'DGI'].values[0] for ligand in ligands]
    else:
        dg_fep = [df_for_plot.loc[df_for_plot['Ligand_Name'] == ligand, column].values[0] for ligand in ligands]

    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    if column == "all":
        ax.scatter(dg_exp, dg_fep_a, label='8FSB', color='C1', alpha=1, edgecolors='none')
        ax.scatter(dg_exp, dg_fep_b, label='6W1Y', color='C0', alpha=1, edgecolors='none')
        for dg_exp_val, dg_fep_a_val, dg_fep_b_val in zip(dg_exp, dg_fep_a, dg_fep_b):
            ax.plot([dg_exp_val, dg_exp_val], [dg_fep_a_val, dg_fep_b_val], color='gray', alpha=0.5, linewidth=0.5, zorder=0)
            if dg_fep_a_val < dg_fep_b_val:
                ax.scatter(dg_exp_val, dg_fep_a_val, color='C1', edgecolors='black', s=50)
            else:
                ax.scatter(dg_exp_val, dg_fep_b_val, color='C0', edgecolors='black', s=50)
        ax.legend(loc='lower right', fontsize=12)
    else:
        ax.scatter(dg_exp, dg_fep)

    ax.set_xlim(-16.5, -7.5)
    ax.set_ylim(-16.5, -7.5)
    ax.set_xlabel(r'Experimental $\Delta G$ (kcal/mol)')
    ax.set_ylabel(r'AB-FEP $\Delta G$ (kcal/mol)')

    def linear_func(x, b):
        return x + b

    popt, pcov = curve_fit(linear_func, dg_exp, dg_fep)
    print(popt)
    ax.plot([-17, -7], [-17, -7], '-', color='black', alpha=1, linewidth=1, zorder=-2)
    x_vals = np.array(ax.get_xlim())
    y_vals = linear_func(x_vals, *popt)
    ax.plot(x_vals, y_vals, '--', color='black', zorder=-1)
    arrow_start = np.array([-14.25, -14.25])
    arrow_end = linear_func(arrow_start[0], *popt)
    ax.annotate('', xy=(arrow_start[0], arrow_end), xytext=(arrow_start[0], arrow_start[1]), arrowprops=dict(arrowstyle='->', color='C2', lw=1), zorder=-1)
    ax.text(arrow_start[0], arrow_start[1], f'offset:\n{popt[0]:.2f}\nkcal/mol', fontsize=10, color='C2', rotation=0, zorder=-1, verticalalignment='bottom', horizontalalignment='right')
    ax.fill_between(x_vals, y_vals - 1, y_vals + 1, color='black', alpha=0.1, edgecolor='none', zorder=-1)
    ax.fill_between(x_vals, y_vals - 2, y_vals + 2, color='black', alpha=0.1, edgecolor='none', zorder=-1)

    rmse = np.sqrt(mean_squared_error(dg_fep, linear_func(np.array(dg_exp), *popt)))
    ax.text(0.05, 0.95, f'RMSE = {rmse:.2f} kcal/mol', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    print(f'RMSE = {rmse:.2f} kcal/mol')
    residuals = np.array(dg_fep) - linear_func(np.array(dg_exp), *popt)
    largest_outlier_index = np.argmax(np.abs(residuals))
    print(f'Largest outlier: {ligands[largest_outlier_index]} ({residuals[largest_outlier_index]:.2f} kcal/mol)')
    r2 = r2_score(dg_fep, linear_func(np.array(dg_exp), *popt))
    ax.text(0.05, 0.875, f'$R^2$ = {r2:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    print(f'$R^2$ = {r2:.2f}')
    fig.tight_layout()
    if save_as is not None:
        fig.savefig(save_as + '.png', dpi=dpi, format='png')
        fig.savefig(save_as + '.pdf', dpi=dpi, format='pdf')
    return fig, ax


# Use the shared minimum dG values for each ligand in the binding-affinity plots.
df_for_plot = data_htr3a.copy()
df_for_plot['MIN(DG)'] = df_for_plot[['DGA', 'DGI']].min(axis=1)
df_for_plot


fig_all, ax = plot_dg_comparison(df_for_plot, ki, column="all", save_as='../plots/htr3a_dg_scatter_all')
#fig_min, ax = plot_dg_comparison(df_for_plot, ki, column="MIN(DG)", save_as='../plots/htr3a_dg_scatter_min')
fig_a, ax = plot_dg_comparison(df_for_plot, ki, column="DGA", save_as='../plots/htr3a_dg_scatter_8fsb')
fig_b, ax = plot_dg_comparison(df_for_plot, ki, column="DGI", save_as='../plots/htr3a_dg_scatter_6w1y')
