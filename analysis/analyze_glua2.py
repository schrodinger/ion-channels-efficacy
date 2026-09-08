# Generated from Analysis-GluA2.ipynb.
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


# =============================
# Functions and Definitions
# =============================


ligand_labels_data = pd.read_csv('../labels/glua2_labels.csv')
ligand_labels = ligand_labels_data.set_index('ligand')['label'].to_dict()


def classification_plot(
        data, score_name='DDG', score_label='$\Delta\Delta$G [kcal/mol]',
        use_index=None, ylim=None,
        bar_color = {'I':'C0', 'A':'C1'},
        bar_label = {'I':'inactive', 'A':'active'},
        figsize=[4,3], dpi=300,
        file_name=None,
        legend_loc='best',
        xlabel_mode='ligand', 
        xlabel_rotation=45
    ):
    
    data = data[data['label']!='-'].sort_values(score_name)
    bar_label_set = {key: False for key in bar_label}
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    plot_labels = []
    for new_index, line in enumerate(data[['ligand','label',score_name]].itertuples()):
        index, ligand, lig_label, ddg = line  
        plot_labels.append(lig_label)
        if bar_label_set[lig_label]:
            blabel = None
        else:
            blabel = bar_label[lig_label]
            bar_label_set[lig_label] = True
        # Define how the x axis will be labeled
        if use_index is None:
            x_value = ligand
        elif use_index == 'original':
            x_value = f'%i'%index
        elif use_index == 'sorted':
            x_value = f'%i'%(new_index+1)
        # Plot the bar for this ligand
        ax.bar(x_value, ddg, color=bar_color[lig_label], label=blabel)
        
    #ax.bar(ant['ligand'], ant['DDG'], label='antagonists', color='C0')
    if xlabel_mode == 'ligand':
        plt.xticks(rotation=xlabel_rotation, ha='right')
        ax.tick_params(axis='x', which='major', labelsize=10)
        for i, tick_label in enumerate(ax.get_xticklabels()):
            tick_label.set_color(bar_color[plot_labels[i]])
    elif xlabel_mode == 'none':
        ax.set_xticklabels([])
        ax.set_xlabel('Ligand')

    ax.set_ylabel(score_label)
    if ylim is not None:
        ax.set_ylim(ylim)    

    if legend_loc == 'above':
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=len(bar_label))
    elif legend_loc == 'stacked':
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=1)
    else:
        ax.legend(loc=legend_loc)

    if xlabel_mode != 'ligand':
        plt.tight_layout()

    if file_name is not None:
        fig.savefig(f'{file_name}.png', dpi=dpi, format='png')
        fig.savefig(f'{file_name}.pdf', dpi=dpi, format='pdf')

    return fig, ax


# ===========================
# Binders vs. Non-Binders
# ===========================


data_i_both = pd.concat([pd.read_csv(file) for file in ['../results/glua2-i_no-restraints.csv', '../results/glua2-i_bb-restraints.csv']])

min_values_i_both = []

for ligand in ligand_labels:
    values_i_both = data_i_both[data_i_both['ligand_name'] == ligand]['pred_dg']
    min_values_i_both.append(np.min(values_i_both))

min_values_i_both = np.nan_to_num(min_values_i_both, nan=0.0)
min_values_i_both[min_values_i_both > 0] = 0

glua2_i_both_columns = ['ligand', 'label', 'DG I']
data_glua2_i_both = pd.DataFrame(columns=glua2_i_both_columns)
data_glua2_i_both['ligand'] = ligand_labels.keys()
data_glua2_i_both['label'] = ligand_labels.values()
data_glua2_i_both['DG I'] = min_values_i_both


data_glua2_i_both['label'] = data_glua2_i_both['label'].replace({'A':'B', 'I':'B'})


display(data_glua2_i_both.sort_values('DG I'))


_,_ = classification_plot(
    data_glua2_i_both, score_name='DG I', score_label='$\\Delta G$ [kcal/mol]',
    use_index=None, ylim=[-9, 0], figsize=[4,2.2], legend_loc='best',
    bar_color={'B':'C2', 'N':'C3'}, bar_label={'B':'competitive', 'N':'non-comp.'},
    file_name='../plots/glua2_barplot_binding'
)


# ============================
# Agonists vs. Antagonists
# ============================


data_a = pd.read_csv('../results/glua2-a_bb-restraints.csv')
data_i = pd.read_csv('../results/glua2-i_bb-restraints.csv')


# Only ligands classified as active or inactive
binding_ligand_labels = {
    ligand: label
    for ligand, label in ligand_labels.items()
    if label != 'N'
}


min_values_a = []
min_values_i = []

for ligand in ligand_labels:
    values_a = data_a[data_a['ligand_name'] == ligand]['pred_dg']
    values_i = data_i[data_i['ligand_name'] == ligand]['pred_dg']

    min_values_a.append(np.min(values_a))
    min_values_i.append(np.min(values_i))
    
    ddg = np.min(values_a) - np.min(values_i)
    
    print("%25a" % ligand, "%1.1f" % np.min(values_a), "%1.1f" % np.min(values_i), "%   1.1f" % ddg)
    
# Replace NaNs with 0.
min_values_a = np.nan_to_num(min_values_a, nan=0.0)
min_values_i = np.nan_to_num(min_values_i, nan=0.0)

# Replace all positive values with 0.
min_values_a[min_values_a > 0] = 0
min_values_i[min_values_i > 0] = 0


ddg_col = 'Absolute Binding free energy'
glua2_columns = ['ligand', 'label', 'DG A', 'DG I', 'DDG']
data_glua2 = pd.DataFrame(columns=glua2_columns)

data_glua2['ligand'] = ligand_labels.keys()
data_glua2['label'] = ligand_labels.values()
data_glua2['DG A'] = min_values_a
data_glua2['DG I'] = min_values_i
data_glua2['DDG'] = min_values_a - min_values_i


_,_ = classification_plot(
    data_glua2[data_glua2['ligand'].isin(binding_ligand_labels)],
    ylim=[-12,12], figsize=[4,2.2],
    file_name='../plots/glua2_barplot_function'
)


_,_ = classification_plot(
    data_glua2_i_both, score_name='DG I', score_label='$\\Delta G$ [kcal/mol]',
    use_index=None, ylim=[-9, 0], figsize=[2,3], legend_loc='stacked', xlabel_mode='none',
    bar_color={'B':'C2', 'N':'C3'}, bar_label={'B':'competitive', 'N':'non-comp.'},
    file_name='../plots/glua2_barplot_binding_stacked'
)


_,_ = classification_plot(
    data_glua2[data_glua2['ligand'].isin(binding_ligand_labels)],
    ylim=[-12,12], figsize=[2,3], legend_loc='stacked', xlabel_mode='none',
    file_name='../plots/glua2_barplot_function_stacked'
)
