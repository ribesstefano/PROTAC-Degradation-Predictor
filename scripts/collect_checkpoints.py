import os
import sys
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    # Create a directory called models if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    trim_str = 'logs_Active_Dmax_0.6_pDC50_6.0_test_split_0.1_random_random_cv_model_fold0/logs_Active_Dmax_0.6_pDC50_6.0_test_split_0.1_'
    # Recursively loop in the log directory
    # in the checkpoint directory, copy the file ending with .ckpt into models but rename its name from 'protac' to its
    # parent directory name removing 'logs_Active_Dmax_0.6_pDC50_6.0_test_split_0.1_random_random_cv_model_fold0/logs_Active_Dmax_0.6_pDC50_6.0_test_split_0.1_'
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.ckpt'):
                checkpoint_file = os.path.join(root, file)
                model_name = file.split(trim_str)[-1]
                if 'tanimoto' in root:
                    split_type = 'tanimoto'
                elif 'random' in root:
                    split_type = 'random'
                elif 'uniprot' in root:
                    split_type = 'uniprot'
                else:
                    raise ValueError('Unknown split type')
                if 'fold' in root:
                    # Get fold number, i.e., subsequent character after fold
                    fold = root.split('fold')[-1][0]
                    # print(f'models/{model_name.replace("protac", f"cv_model_{split_type}_fold{fold}")}')
                    model_name = model_name.replace("protac", f"cv_model_{split_type}_fold{fold}")
                else:
                    model_name = model_name.replace("val_", "test_")
                
                # Check if in destination directory, there is a "similar" model
                base_model_name = model_name.split('-')[0]
                old_model_name = None
                # Loop over the models directory to check if there is a similar model
                for model in os.listdir(models_dir):
                    if base_model_name in model:
                        old_model_name = model
                        break
                # Now check the accuracy and ROC-AUC of the old model and the new model
                # If the new model is better, then replace the old model with the new model
                # Example of model name: cv_model_random_fold0-epoch=00-val_acc=0.00-val_roc_auc=0.00.ckpt
                if old_model_name is not None:
                    if 'val_acc' in model_name:
                        old_acc = float(re.search(r'val_acc=(\d+\.\d+)', old_model_name).group(1))
                        old_roc_auc = float(re.search(r'val_roc_auc=(\d+\.\d+)', old_model_name).group(1))
                        new_acc = float(re.search(r'val_acc=(\d+\.\d+)', model_name).group(1))
                        new_roc_auc = float(re.search(r'val_roc_auc=(\d+\.\d+)', model_name).group(1))
                        if new_acc > old_acc and new_roc_auc > old_roc_auc:
                            print(f'Replacing {old_model_name} with {model_name}')
                            os.system(f'rm {os.path.join(models_dir, old_model_name)}')
                            os.system(f'cp {checkpoint_file} {os.path.join(models_dir, model_name)}')
                    if 'test_acc' in model_name:
                        old_acc = float(re.search(r'test_acc=(\d+\.\d+)', old_model_name).group(1))
                        old_roc_auc = float(re.search(r'test_roc_auc=(\d+\.\d+)', old_model_name).group(1))
                        new_acc = float(re.search(r'test_acc=(\d+\.\d+)', model_name).group(1))
                        new_roc_auc = float(re.search(r'test_roc_auc=(\d+\.\d+)', model_name).group(1))
                        if new_acc > old_acc and new_roc_auc > old_roc_auc:
                            print(f'Replacing {old_model_name} with {model_name}')
                            os.system(f'rm {os.path.join(models_dir, old_model_name)}')
                            os.system(f'cp {checkpoint_file} {os.path.join(models_dir, model_name)}')
                else:
                    print(f'Copying {model_name}')
                    os.system(f'cp {checkpoint_file} {os.path.join(models_dir, model_name)}')

if __name__ == '__main__':
    main()