# Generated from Analysis-a3b4-nAChR.ipynb.
# The working directory is set to this script's directory so ../ paths resolve to the repository root.
from pathlib import Path as _NotebookPath
import os as _notebook_os
_notebook_os.chdir(_NotebookPath(__file__).resolve().parent)
_notebook_os.makedirs("../plots", exist_ok=True)  # output folder for all figures

try:
    from IPython.display import display
except ImportError:
    display = print

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from analysis_utils import combine_dataframes, ddg_barplot



# Load labels and alternative names
labels_data = pd.read_csv('../labels/a3b4-nachr_labels.csv')
names = labels_data.set_index('ligand')['name'].to_dict()
labels = labels_data.set_index('ligand')['label'].to_dict()

# Studies: label -> (active results, inactive results)
studies = {
    'default-restraints': ('../results/a3b4-nachr-a.csv', '../results/a3b4-nachr-i.csv'),
    'no-restraints': ('../results/a3b4-nachr-a_no-restraints.csv', '../results/a3b4-nachr-i_no-restraints.csv'),
    'weak-restraints-bw00-f01': ('../results/a3b4-nachr-a_restraints-bw00-f01.csv', '../results/a3b4-nachr-i_restraints-bw00-f01.csv'),
}

metrics = []
merged_by_study = {}

for study, (file_a, file_i) in studies.items():
    # Load AB-FEP results
    df_a = pd.read_csv(file_a)
    df_i = pd.read_csv(file_i)

    # Merge AB-FEP results and sort by DDG
    merged_df = combine_dataframes(df_a, df_i)
    merged_df = merged_df[merged_df['Ligand_Name'] != '18-MC']  # decoy ligand from the main study
    merged_df = merged_df.sort_values(by='DDG', ascending=True)
    print(study)
    display(merged_df)

    _,_ = ddg_barplot(
        merged_df, labels, figsize=(4, 3.5), errorbars=False, xlabel_mode='dict', ligand_names=names, ylim=(-3, 7),
        file_name=f'../plots/a3b4-nachr_barplot_ligand-names_{study}'
    )

    merged_by_study[study] = merged_df

    # Classification metrics: agonist (positive) vs antagonist.
    # Lower DDG means more agonist-like, so the score is -DDG; absolute accuracy uses DDG < 0 as agonist.
    classified = merged_df[merged_df['Ligand_Name'].map(labels).isin(['agonist', 'antagonist'])]
    y_true = (classified['Ligand_Name'].map(labels) == 'agonist').astype(int).values
    auc = roc_auc_score(y_true, -classified['DDG'].values)
    accuracy = ((classified['DDG'].values < 0).astype(int) == y_true).mean()
    metrics.append({'study': study, 'n': len(classified), 'AUC-ROC': auc, 'accuracy': accuracy})

metrics_df = pd.DataFrame(metrics)
display(metrics_df)


# Joint figure: one panel per restraint force constant, shared y axis.
force_constants = {
    'default-restraints': r'$\mathbfit{f}_\mathbfit{c}$ = 10 kcal/mol',
    'weak-restraints-bw00-f01': r'$\mathbfit{f}_\mathbfit{c}$ = 1 kcal/mol',
    'no-restraints': r'$\mathbfit{f}_\mathbfit{c}$ = 0 (no restraints)',
}
auc_by_study = metrics_df.set_index('study')['AUC-ROC'].to_dict()
accuracy_by_study = metrics_df.set_index('study')['accuracy'].to_dict()

fig, axes = plt.subplots(3, 1, figsize=(7.2, 9), dpi=300, sharex=True)
for ax, (study, fc_text) in zip(axes, force_constants.items()):
    data = merged_by_study[study]
    data = data[data['Ligand_Name'].map(labels).isin(['agonist', 'antagonist'])]
    colors = ['C1' if labels[lig] == 'agonist' else 'C0' for lig in data['Ligand_Name']]
    y = np.arange(len(data))
    ax.barh(y, data['DDG'].values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([names[lig] for lig in data['Ligand_Name']], fontsize=10)
    ax.invert_yaxis()  # most negative DDG at the top
    ax.set_xlim(-3, 7)
    # Force constant in bold; metrics in regular weight below it (leading blank line adds spacing).
    ax.text(0.98, 0.96, fc_text, transform=ax.transAxes, ha='right', va='top', fontsize=12, fontweight='bold')
    ax.text(
        0.98, 0.96, f"\n\nAUC-ROC = {auc_by_study[study]:.2f}\nAccuracy = {accuracy_by_study[study]:.2f}",
        transform=ax.transAxes, ha='right', va='top', fontsize=12,
    )
axes[-1].set_xlabel(r'$\Delta\Delta G$ [kcal/mol]', size=12)
fig.tight_layout()
fig.savefig('../plots/a3b4-nachr_barplot_ligand-names_restraints-comparison.png', dpi=300, format='png')
fig.savefig('../plots/a3b4-nachr_barplot_ligand-names_restraints-comparison.pdf', dpi=300, format='pdf')
