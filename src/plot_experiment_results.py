import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


palette = ['#83B8FE', '#FFA54C', '#94ED67', '#FF7FFF']


def plot_training_curves(df, split_type, stage='test', multimodels=False, groupby='model_id'):
    Stage = 'Test' if stage == 'test' else 'Validation'

    # Clean the data
    df = df.dropna(how='all', axis=1)

    # Convert all columns to numeric, setting errors='coerce' to handle non-numeric data
    df = df.apply(pd.to_numeric, errors='coerce')

    # Group by 'epoch' and aggregate by mean
    if multimodels:
        epoch_data = df.groupby([groupby, 'epoch']).mean().reset_index()
    else:
        epoch_data = df.groupby('epoch').mean().reset_index()

    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    # Plot training loss
    # ax[0].plot(epoch_data.index, epoch_data['train_loss_epoch'], label='Training Loss')
    # ax[0].plot(epoch_data.index, epoch_data[f'{stage}_loss'], label=f'{Stage} Loss', linestyle='--')
    sns.lineplot(data=epoch_data, x='epoch', y='train_loss_epoch', ax=ax[0], label='Training Loss')
    sns.lineplot(data=epoch_data, x='epoch', y=f'{stage}_loss', ax=ax[0], label=f'{Stage} Loss', linestyle='--')

    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='lower right')
    ax[0].grid(axis='both', alpha=0.5)

    # Plot training accuracy
    # ax[1].plot(epoch_data.index, epoch_data['train_acc_epoch'], label='Training Accuracy')
    # ax[1].plot(epoch_data.index, epoch_data[f'{stage}_acc'], label=f'{Stage} Accuracy', linestyle='--')
    sns.lineplot(data=epoch_data, x='epoch', y='train_acc_epoch', ax=ax[1], label='Training Accuracy')
    sns.lineplot(data=epoch_data, x='epoch', y=f'{stage}_acc', ax=ax[1], label=f'{Stage} Accuracy', linestyle='--')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend(loc='lower right')
    ax[1].grid(axis='both', alpha=0.5)
    # Set limit to y-axis
    ax[1].set_ylim(0, 1.0)
    # Set y-axis to percentage
    ax[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))

    # Plot training ROC-AUC
    # ax[2].plot(epoch_data.index, epoch_data['train_roc_auc_epoch'], label='Training ROC-AUC')
    # ax[2].plot(epoch_data.index, epoch_data[f'{stage}_roc_auc'], label=f'{Stage} ROC-AUC', linestyle='--')
    sns.lineplot(data=epoch_data, x='epoch', y='train_roc_auc_epoch', ax=ax[2], label='Training ROC-AUC')
    sns.lineplot(data=epoch_data, x='epoch', y=f'{stage}_roc_auc', ax=ax[2], label=f'{Stage} ROC-AUC', linestyle='--')
    ax[2].set_ylabel('ROC-AUC')
    ax[2].legend(loc='lower right')
    ax[2].grid(axis='both', alpha=0.5)
    # Set limit to y-axis
    ax[2].set_ylim(0, 1.0)
    # Set x-axis label
    ax[2].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(f'plots/training_metrics_{split_type}.pdf', bbox_inches='tight')
    

def plot_performance_metrics(df_cv, df_test, title=None):

    # Extract and prepare CV data
    cols = ['model_type', 'fold', 'val_acc', 'val_roc_auc', 'split_type']
    if 'test_acc' in df_cv.columns:
        cols.extend(['test_acc', 'test_roc_auc'])
    cv_data = df_cv[cols]
    cv_data = cv_data.melt(id_vars=['model_type', 'fold', 'split_type'], var_name='Metric', value_name='Score')
    cv_data['Metric'] = cv_data['Metric'].replace({
        'val_acc': 'Validation Accuracy',
        'val_roc_auc': 'Validation ROC AUC',
        'test_acc': 'Test Accuracy',
        'test_roc_auc': 'Test ROC AUC'
    })
    cv_data['Stage'] = cv_data['Metric'].apply(lambda x: 'Validation' if 'Val' in x else 'Test')
    # Remove test data from CV data
    cv_data = cv_data[cv_data['Stage'] == 'Validation']

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
    sns.barplot(
        data=combined_data,
        x='Metric',
        y='Score',
        hue='Split Type',
        errorbar=('sd', 1),
        palette=palette)
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
            value = f'{p.get_height():.1%}'
        else:
            value = f'{p.get_height():.3f}'
        
        print(f'Plotting value: {p.get_height()} -> {value}')
        x = p.get_x() + p.get_width() / 2
        y = 0.4 # p.get_height() - p.get_height() / 2
        plt.annotate(value, (x, y), ha='center', va='center', color='black', fontsize=10, rotation=90, alpha=0.8)

    plt.savefig(f'plots/{title}.pdf', bbox_inches='tight')


def plot_ablation_study(report, title=''):
    # Define the ablation study combinations
    ablation_study_combinations = [
        'disabled smiles',
        'disabled poi',
        'disabled e3',
        'disabled cell',
        'disabled poi e3',
        'disabled poi e3 smiles',
        'disabled poi e3 cell',
    ]

    for group in report['split_type'].unique():    
        baseline = report[report['disabled_embeddings'].isna()].copy()
        baseline = baseline[baseline['split_type'] == group]
        baseline['disabled_embeddings'] = 'all embeddings enabled'
        # metrics_to_show = ['val_acc', 'test_acc']
        metrics_to_show = ['test_acc']
        # baseline = baseline.melt(id_vars=['fold', 'disabled_embeddings'], value_vars=metrics_to_show, var_name='metric', value_name='score')
        baseline = baseline.melt(id_vars=['disabled_embeddings'], value_vars=metrics_to_show, var_name='metric', value_name='score')

        print('baseline:\n', baseline)

        ablation_dfs = []
        for disabled_embeddings in ablation_study_combinations:
            tmp = report[report['disabled_embeddings'] == disabled_embeddings].copy()
            tmp = tmp[tmp['split_type'] == group]
            # tmp = tmp.melt(id_vars=['fold', 'disabled_embeddings'], value_vars=metrics_to_show, var_name='metric', value_name='score')
            tmp = tmp.melt(id_vars=['disabled_embeddings'], value_vars=metrics_to_show, var_name='metric', value_name='score')
            ablation_dfs.append(tmp)
        ablation_df = pd.concat(ablation_dfs)

        print('ablation_df:\n', ablation_df)

        # dummy_val_df = pd.DataFrame()
        # tmp = report[report['split_type'] == group]
        # dummy_val_df['score'] = tmp[['val_active_perc', 'val_inactive_perc']].max(axis=1)
        # dummy_val_df['metric'] = 'val_acc'
        # dummy_val_df['disabled_embeddings'] = 'dummy'

        dummy_test_df = pd.DataFrame()
        tmp = report[report['split_type'] == group]
        dummy_test_df['score'] = tmp[['test_active_perc', 'test_inactive_perc']].max(axis=1)
        dummy_test_df['metric'] = 'test_acc'
        dummy_test_df['disabled_embeddings'] = 'dummy'

        # dummy_df = pd.concat([dummy_val_df, dummy_test_df])
        dummy_df = dummy_test_df

        final_df = pd.concat([dummy_df, baseline, ablation_df])

        final_df['metric'] = final_df['metric'].map({
            'val_acc': 'Validation Accuracy',
            'test_acc': 'Test Accuracy',
            'val_roc_auc': 'Val ROC-AUC',
            'test_roc_auc': 'Test ROC-AUC',
        })

        final_df['disabled_embeddings'] = final_df['disabled_embeddings'].map({
            'all embeddings enabled': 'All embeddings enabled',
            'dummy': 'Dummy model',
            'disabled smiles': 'Disabled compound information',
            'disabled e3': 'Disabled E3 information',
            'disabled poi': 'Disabled target information',
            'disabled cell': 'Disabled cell information',
            'disabled poi e3': 'Disabled E3 and target info',
            'disabled poi e3 smiles': 'Disabled compound, E3, and target info\n(only cell information left)',
            'disabled poi e3 cell': 'Disabled cell, E3, and target info\n(only compound information left)',
        })

        # Print final_df to latex
        tmp  = final_df.groupby(['disabled_embeddings', 'metric']).mean().round(3)
        # Remove fold column to tmp
        tmp = tmp.reset_index() #.drop('fold', axis=1)

        print('DF to plot:\n', tmp.to_markdown(index=False))

        # fig, ax = plt.subplots(figsize=(5, 5))
        fig, ax = plt.subplots()

        sns.barplot(data=final_df,
            y='disabled_embeddings',
            x='score',
            hue='metric',
            ax=ax,
            errorbar=('sd', 1),
            palette=sns.color_palette(palette, len(palette)),
            saturation=1,
        )

        # ax.set_title(f'{group.replace("random", "standard")} CV split')
        ax.grid(axis='x', alpha=0.5)
        ax.tick_params(axis='y', rotation=0)
        ax.set_xlim(0, 1.0)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.set_ylabel('')
        ax.set_xlabel('')
        # Set the legend outside the plot and below
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)
        # Set the legend in the upper right corner
        ax.legend(loc='upper right')

        # For each bar, add the rotated value (as percentage), inside the bar
        for i, p in enumerate(plt.gca().patches):
            # TODO: For some reasons, there is an additional bar being added at
            # the end of the plot... it's not in the dataframe
            if i == len(plt.gca().patches) - 1:
                continue
            value = '{:.1f}%'.format(100 * p.get_width())
            y = p.get_y() + p.get_height() / 2
            x = 0.4 # p.get_height() - p.get_height() / 2
            plt.annotate(value, (x, y), ha='center', va='center', color='black', fontsize=10, alpha=0.8)

        plt.savefig(f'plots/{title}{group}.pdf', bbox_inches='tight')


def plot_majority_voting_performance(df):
    # cv_models,test_acc,test_roc_auc,split_type
    # Melt the dataframe
    df = df.melt(id_vars=['cv_models', 'test_acc', 'test_roc_auc', 'split_type'], var_name='Metric', value_name='Score')
    print(df)


def main():
    active_col = 'Active (Dmax 0.6, pDC50 6.0)'
    test_split = 0.1
    n_models_for_test = 3
    cv_n_folds = 5

    active_name = active_col.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    dataset_info = f'{active_name}_test_split_{test_split}'

    # Load the data
    reports = {}
    for experiment in ['', 'xgboost_', 'cellsonehot_', 'aminoacidcnt_']:
        reports[f'{experiment}cv_train'] = pd.concat([
            pd.read_csv(f'reports/{experiment}cv_report_{dataset_info}_standard.csv'),
            pd.read_csv(f'reports/{experiment}cv_report_{dataset_info}_target.csv'),
            pd.read_csv(f'reports/{experiment}cv_report_{dataset_info}_similarity.csv'),
        ])
        reports[f'{experiment}test'] = pd.concat([
            pd.read_csv(f'reports/{experiment}test_report_{dataset_info}_standard.csv'),
            pd.read_csv(f'reports/{experiment}test_report_{dataset_info}_target.csv'),
            pd.read_csv(f'reports/{experiment}test_report_{dataset_info}_similarity.csv'),
        ])
        reports[f'{experiment}hparam'] = pd.concat([
            pd.read_csv(f'reports/{experiment}hparam_report_{dataset_info}_standard.csv'),
            pd.read_csv(f'reports/{experiment}hparam_report_{dataset_info}_target.csv'),
            pd.read_csv(f'reports/{experiment}hparam_report_{dataset_info}_similarity.csv'),
        ])
        reports[f'{experiment}majority_vote'] = pd.concat([
            pd.read_csv(f'reports/{experiment}majority_vote_report_{dataset_info}_standard.csv'),
            pd.read_csv(f'reports/{experiment}majority_vote_report_{dataset_info}_target.csv'),
            pd.read_csv(f'reports/{experiment}majority_vote_report_{dataset_info}_similarity.csv'),
        ])
        if experiment != 'xgboost_':
            reports[f'{experiment}ablation'] = pd.concat([
                pd.read_csv(f'reports/{experiment}ablation_report_{dataset_info}_standard.csv'),
                pd.read_csv(f'reports/{experiment}ablation_report_{dataset_info}_target.csv'),
                pd.read_csv(f'reports/{experiment}ablation_report_{dataset_info}_similarity.csv'),
            ])

    for experiment in ['', 'xgboost_', 'cellsonehot_', 'aminoacidcnt_']:
        print('=' * 80)
        print(f'Experiment: {experiment}')
        print('=' * 80)

        # Plot training curves
        for split_type in ['standard', 'similarity', 'target']:
            # Skip XGBoost: we don't have its training curves
            if experiment != 'xgboost_':
                # Plot training curves for the best models
                split_metrics = []
                for i in range(n_models_for_test):
                    metrics_dir = f'best_model_n{i}_{experiment}{split_type}_{dataset_info}'
                    metrics = pd.read_csv(f'logs/{metrics_dir}/{metrics_dir}/metrics.csv')
                    metrics['model_id'] = i
                    # Rename 'val_' columns to 'test_' columns
                    metrics = metrics.rename(columns={'val_loss': 'test_loss', 'val_acc': 'test_acc', 'val_roc_auc': 'test_roc_auc'})
                    split_metrics.append(metrics)
                plot_training_curves(pd.concat(split_metrics), f'{experiment}{split_type}_best_model', multimodels=True)

                # Plot training curves for the CV models
                split_metrics_cv = []
                for i in range(cv_n_folds):
                    metrics_dir = f'cv_model_{experiment}{split_type}_{dataset_info}_fold{i}'
                    metrics = pd.read_csv(f'logs/{metrics_dir}/{metrics_dir}/metrics.csv')
                    metrics['fold'] = i
                    split_metrics_cv.append(metrics)
                plot_training_curves(pd.concat(split_metrics_cv), f'{experiment}{split_type}_cv_model', stage='val', multimodels=True, groupby='fold')

        if experiment != 'xgboost_':
            # Skip XGBoost: we don't have test data for its CV models
            plot_performance_metrics(
                reports[f'{experiment}cv_train'],
                reports[f'{experiment}cv_train'],
                title=f'{experiment}mean_performance-cv_models_as_test',
            )
            plot_performance_metrics(
                reports[f'{experiment}cv_train'],
                reports[f'{experiment}majority_vote'][reports[f'{experiment}majority_vote']['cv_models'] == True],
                title=f'{experiment}majority_vote_performance-cv_models_as_test',
            )
            # Skip XGBoost: we don't have its ablation study
            reports[f'{experiment}test']['disabled_embeddings'] = pd.NA
            plot_ablation_study(
                    pd.concat([
                    reports[f'{experiment}ablation'],
                    reports[f'{experiment}test'],
                ]),
                title=f'{experiment}ablation_study_',
            )

        plot_performance_metrics(
            reports[f'{experiment}cv_train'],
            reports[f'{experiment}test'],
            title=f'{experiment}mean_performance-best_models_as_test',
        )

        # 
        if experiment == 'xgboost_':
            df_test = reports[f'{experiment}majority_vote']
        else:
            df_test = reports[f'{experiment}majority_vote'][reports[f'{experiment}majority_vote']['cv_models'].isna()]
        plot_performance_metrics(
            reports[f'{experiment}cv_train'],
            df_test,
            title=f'{experiment}majority_vote_performance-best_models_as_test',
        )

        # # Plot hyperparameter optimization results to markdown
        # print(reports['hparam'][['split_type', 'hidden_dim', 'learning_rate', 'dropout', 'use_smote', 'smote_k_neighbors']].to_markdown(index=False))


if __name__ == '__main__':
    main()