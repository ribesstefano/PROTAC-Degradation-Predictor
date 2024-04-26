import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


palette = ['#83B8FE', '#FFA54C', '#94ED67', '#FF7FFF']


def plot_metrics(df, title):
    # Clean the data
    df = df.dropna(how='all', axis=1)

    # Convert all columns to numeric, setting errors='coerce' to handle non-numeric data
    df = df.apply(pd.to_numeric, errors='coerce')

    # Group by 'epoch' and aggregate by mean
    epoch_data = df.groupby('epoch').mean()

    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    # Plot training loss
    ax[0].plot(epoch_data.index, epoch_data['train_loss_epoch'], label='Training Loss')
    ax[0].plot(epoch_data.index, epoch_data['test_loss'], label='Test Loss', linestyle='--')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='lower right')
    ax[0].grid(axis='both', alpha=0.5)

    # Plot training accuracy
    ax[1].plot(epoch_data.index, epoch_data['train_acc_epoch'], label='Training Accuracy')
    ax[1].plot(epoch_data.index, epoch_data['test_acc'], label='Test Accuracy', linestyle='--')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend(loc='lower right')
    ax[1].grid(axis='both', alpha=0.5)

    # Plot training ROC-AUC
    ax[2].plot(epoch_data.index, epoch_data['train_roc_auc_epoch'], label='Training ROC-AUC')
    ax[2].plot(epoch_data.index, epoch_data['test_roc_auc'], label='Test ROC-AUC', linestyle='--')
    ax[2].set_ylabel('ROC-AUC')
    ax[2].legend(loc='lower right')
    ax[2].grid(axis='both', alpha=0.5)

    # Set x-axis label
    ax[2].set_xlabel('Epoch')

    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'plots/{title}_metrics.pdf', bbox_inches='tight')
    

def plot_report(df_cv, df_test, title=None):

    # Extract and prepare CV data
    cv_data = df_cv[['model_type', 'fold', 'val_acc', 'val_roc_auc', 'test_acc', 'test_roc_auc', 'split_type']]
    cv_data = cv_data.melt(id_vars=['model_type', 'fold', 'split_type'], var_name='Metric', value_name='Score')
    cv_data['Metric'] = cv_data['Metric'].replace({
        'val_acc': 'Validation Accuracy',
        'val_roc_auc': 'Validation ROC AUC',
        'test_acc': 'Test Accuracy',
        'test_roc_auc': 'Test ROC AUC'
    })
    cv_data['Stage'] = cv_data['Metric'].apply(lambda x: 'Validation' if 'Val' in x else 'Test')

    # Extract and prepare test data
    test_data = df_test[['model_type', 'test_acc', 'test_roc_auc', 'split_type']]
    test_data = test_data.melt(id_vars=['model_type', 'split_type'], var_name='Metric', value_name='Score')
    test_data['Metric'] = test_data['Metric'].replace({
        'test_acc': 'Test Accuracy',
        'test_roc_auc': 'Test ROC AUC'
    })
    test_data['Stage'] = 'Test'

    # Combine CV and test data
    combined_data = pd.concat([cv_data, test_data], ignore_index=True)

    # Rename 'split_type' values according to a predefined map for clarity
    group2name = {
        'random': 'Standard Split',
        'uniprot': 'Target Split',
        'tanimoto': 'Similarity Split',
    }
    combined_data['Split Type'] = combined_data['split_type'].map(group2name)

    # Add dummy model data
    dummy_val_acc = []
    dummy_test_acc = []
    for i, group in enumerate(group2name.keys()):
        # Get the majority class in group_df
        group_df = df_cv[df_cv['split_type'] == group]
        major_col = 'inactive' if group_df['val_inactive_perc'].mean() > 0.5 else 'active'
        dummy_val_acc.append(group_df[f'val_{major_col}_perc'].mean())

        group_df = df_test[df_test['split_type'] == group]
        major_col = 'inactive' if group_df['test_inactive_perc'].mean() > 0.5 else 'active'
        dummy_test_acc.append(group_df[f'test_{major_col}_perc'].mean())

    dummy_scores = []
    metrics = ['Validation Accuracy', 'Validation ROC AUC', 'Test Accuracy', 'Test ROC AUC']
    for i in range(len(dummy_val_acc)):
        for metric, score in zip(metrics, [dummy_val_acc[i], 0.5, dummy_test_acc[i], 0.5]):
            dummy_scores.append({
                'Experiment': i,
                'Metric': metric,
                'Score': score,
                'Split Type': 'Dummy model',
            })
    dummy_model = pd.DataFrame(dummy_scores)
    combined_data = pd.concat([combined_data, dummy_model], ignore_index=True)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(data=combined_data, x='Metric', y='Score', hue='Split Type', errorbar='sd', palette=palette)
    plt.title('')
    plt.ylabel('')
    plt.xlabel('')
    plt.ylim(0, 1.0)  # Assuming scores are normalized between 0 and 1
    plt.grid(axis='y', alpha=0.5, linewidth=0.5)

    # Make the y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    # Plot the legend below the x-axis, outside the plot, and divided in two columns
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4)

    # For each bar, add the rotated value (as percentage), inside the bar
    for i, p in enumerate(plt.gca().patches):
        # TODO: For some reasons, there are 4 additional rectangles being
        # plotted... I suspect it's because the dummy_df doesn't have the same
        # shape as the df containing all the evaluation data...
        if p.get_height() < 0.01:
            continue
        if i % 2 == 0:
            value = '{:.1f}%'.format(100 * p.get_height())
        else:
            value = '{:.2f}'.format(p.get_height())
        
        print(f'Plotting value: {p.get_height()} -> {value}')
        x = p.get_x() + p.get_width() / 2
        y = 0.4 # p.get_height() - p.get_height() / 2
        plt.annotate(value, (x, y), ha='center', va='center', color='black', fontsize=10, rotation=90, alpha=0.8)

    plt.savefig(f'plots/{title}.pdf', bbox_inches='tight')


def main():
    active_col = 'Active (Dmax 0.6, pDC50 6.0)'
    test_split = 0.1
    n_models_for_test = 3

    active_name = active_col.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    report_base_name = f'{active_name}_test_split_{test_split}'

    # Load the data
    reports = {
        'cv_train': pd.read_csv(f'reports/report_cv_train_{report_base_name}.csv'),
        'test': pd.read_csv(f'reports/report_test_{report_base_name}.csv'),
        'ablation': pd.read_csv(f'reports/report_ablation_{report_base_name}.csv'),
        'hparam': pd.read_csv(f'reports/report_hparam_{report_base_name}.csv'),
    }


    # metrics = {}
    # for i in range(n_models_for_test):
    #     for split_type in ['random', 'tanimoto', 'uniprot', 'e3_ligase']:
    #         logs_dir = f'logs_{report_base_name}_{split_type}_best_model_n{i}'
    #         metrics[f'{split_type}_{i}'] = pd.read_csv(f'logs/{logs_dir}/{logs_dir}/metrics.csv')
    #         metrics[f'{split_type}_{i}']['model_id'] = i
    #         # Rename 'val_' columns to 'test_' columns
    #         metrics[f'{split_type}_{i}'] = metrics[f'{split_type}_{i}'].rename(columns={'val_loss': 'test_loss', 'val_acc': 'test_acc', 'val_roc_auc': 'test_roc_auc'})

    #         plot_metrics(metrics[f'{split_type}_{i}'], f'{split_type}_{i}')


    df_val = reports['cv_train']
    df_test = reports['test']
    plot_report(df_val, df_test, title=f'{active_name}_metrics')


if __name__ == '__main__':
    main()