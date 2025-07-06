import argparse

import numpy as np
import pandas as pd
import torch

import protac_degradation_predictor as pdp

def main(args):
    train_df = pd.read_csv(args.train_df)
    val_df = pd.read_csv(args.val_df)
    test_df = pd.read_csv(args.test_df) if args.test_df else None

    # Rename columns to match expected labels
    train_df.rename(columns={
        args.active_label: "Active",
        args.e3_label: "E3 Ligase",
        args.smiles_label: "Smiles",
        args.uniprot_label: "Uniprot",
        args.cell_type_label: "Cell Type"
    }, inplace=True)
    val_df.rename(columns={
        args.active_label: "Active",
        args.e3_label: "E3 Ligase",
        args.smiles_label: "Smiles",
        args.uniprot_label: "Uniprot",
        args.cell_type_label: "Cell Type"
    }, inplace=True)
    if test_df is not None:
        test_df.rename(columns={
            args.active_label: "Active",
            args.e3_label: "E3 Ligase",
            args.smiles_label: "Smiles",
            args.uniprot_label: "Uniprot",
            args.cell_type_label: "Cell Type"
        }, inplace=True)

    # Drop rows with NaN values in critical columns
    # These columns are essential for training and evaluation, so we drop any
    # rows that have NaN values in any of these columns. 
    dropna_subsets = [
        "Active",
        "E3 Ligase",
        "Smiles",
        "Uniprot",
        "Cell Type",
    ]
    train_df = train_df.dropna(subset=dropna_subsets, how="any")
    val_df = val_df.dropna(subset=dropna_subsets, how="any")
    if test_df is not None:
        test_df = test_df.dropna(subset=dropna_subsets, how="any")

    # Get SMILES and precompute fingerprints dictionary
    all_df = [train_df, val_df, test_df]
    all_df = [df for df in all_df if df is not None]  # Filter out None DataF.
    unique_smiles = pd.concat(all_df)["Smiles"].unique().tolist()
    smiles2fp = {
        s: np.array(pdp.get_fingerprint(s), dtype=np.float32)
        for s in unique_smiles
    }
    
    # Get precomputed embeddings
    protein2embedding = pdp.load_protein2embedding()
    cell2embedding = pdp.load_cell2embedding()

    model = None
    if args.finetuning_checkpoint is not None:
        # Load the model from the provided checkpoint for finetuning
        print(f"Loading model from checkpoint: {args.finetuning_checkpoint}")
        model = pdp.load_model(args.finetuning_checkpoint)

    # Train the model
    _, _, metrics = pdp.train_model(
        protein2embedding=protein2embedding,
        cell2embedding=cell2embedding,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        smiles2fp=smiles2fp,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        dropout=args.dropout,
        max_epochs=args.max_epochs,
        use_batch_norm=args.use_batch_norm,
        join_embeddings=args.join_embeddings,
        smote_k_neighbors=args.smote_k_neighbors,
        apply_scaling=args.apply_scaling,
        active_label=args.active_label,
        fast_dev_run=args.fast_dev_run,
        use_logger=not args.disable_logger,
        logger_save_dir=args.logger_save_dir,
        logger_name=args.logger_name,
        logger_version=args.logger_version,
        enable_checkpointing=not args.disable_checkpointing,
        checkpoint_model_name=args.checkpoint_model_name,
        model=model,
    )

    print("Training completed.")

    # Prepare metrics for display
    train_metrics = {k.replace("train_", "").replace("_", " ").title(): v for k, v in metrics.items() if k.startswith("train_")}
    val_metrics = {k.replace("val_", "").replace("_", " ").title(): v for k, v in metrics.items() if k.startswith("val_")}
    test_metrics = {k.replace("test_", "").replace("_", " ").title(): v for k, v in metrics.items() if k.startswith("test_")} if test_df is not None else {}

    # Get a dataframes of metrics with two columns: metric name and value
    train_metrics_df = pd.DataFrame(list(train_metrics.items()), columns=["Metric", "Value"])
    val_metrics_df = pd.DataFrame(list(val_metrics.items()), columns=["Metric", "Value"])
    test_metrics_df = pd.DataFrame(list(test_metrics.items()), columns=["Metric", "Value"])
    
    # Print metrics dataframes as markdown tables
    print("\nTrain Metrics:")
    print("=" * 14)
    print(train_metrics_df.to_markdown(index=False))
    print("\nValidation Metrics:")
    print("=" * 19)
    print(val_metrics_df.to_markdown(index=False))
    if test_df is not None:
        print("\nTest Metrics:")
        print("=" * 13)
        print(test_metrics_df.to_markdown(index=False))

    print("\nTraining completed successfully.")

    if not args.disable_logger:
        logs_dest = f"{args.logger_save_dir}/{args.logger_name}/{args.logger_version}"
        print(f"Logs saved to: {logs_dest}")
        if not args.disable_checkpointing:
            print(f"Model checkpoint saved to folder: {logs_dest}/checkpoints")
    elif not args.disable_checkpointing:
        print(f"Model checkpoint saved to folder: checkpoints")
    
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on custom data.")
    parser.add_argument(
        "--train_df",
        type=str,
        required=True,
        help="Path to the training DataFrame (CSV file).",
    )
    parser.add_argument(
        "--val_df",
        type=str,
        required=True,
        help="Path to the validation DataFrame (CSV file).",
    )
    parser.add_argument(
        "--test_df",
        type=str,
        default=None,
        help="Path to the test DataFrame (CSV file). If not provided, no test set will be used.",
    )
    parser.add_argument(
        "--active_label",
        type=str,
        default="Active",
        help="Label for the active class in the supplied dataframes.",
    )
    parser.add_argument(
        "--e3_label",
        type=str,
        default="E3 Ligase",
        help="Label for the E3 ligase in the supplied dataframes.",
    )
    parser.add_argument(
        "--smiles_label",
        type=str,
        default="Smiles",
        help="Label for the SMILES strings in the supplied dataframes.",
    )
    parser.add_argument(
        "--uniprot_label",
        type=str,
        default="Uniprot",
        help="Label for the Uniprot IDs in the supplied dataframes.",
    )
    parser.add_argument(
        "--cell_type_label",
        type=str,
        default="Cell Type",
        help="Label for the cell type in the supplied dataframes.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=768,
        help="Dimension of the hidden layers in the model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default= 0.999,
        help="Beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Epsilon parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for the model.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum number of epochs for training.",
    )
    parser.add_argument(
        "--use_batch_norm",
        action="store_true",
        help="Whether to use batch normalization in the model.",
    )
    parser.add_argument(
        "--join_embeddings",
        type=str,
        choices=["beginning", "concat", "sum"],
        default="sum",
        help="Method to join embeddings: \"beginning\", \"concat\", or \"sum\".",
    )
    parser.add_argument(
        "--smote_k_neighbors",
        type=int,
        default=5,
        help="Number of neighbors to use in SMOTE for oversampling.",
    )
    parser.add_argument(
        "--apply_scaling",
        action="store_true",
        help="Whether to apply scaling to the features.",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run a quick development run with a small subset of data.",
    )
    parser.add_argument(
        "--disable_logger",
        action="store_true",
        help="Whether to disable logger for training.",
    )
    parser.add_argument(
        "--logger_save_dir",
        type=str,
        default="./logs",
        help="Directory to save the logger outputs.",
    )
    parser.add_argument(
        "--logger_name",
        type=str,
        default="protac-model",
        help="Name of the logger.",
    )
    parser.add_argument(
        "--logger_version",
        type=str,
        default="v0",
        help="Version of the logs. See PyTorch Lightning documentation for more details.",
    )
    parser.add_argument(
        "--disable_checkpointing",
        action="store_true",
        help="Whether to disable model checkpointing during training.",
    )
    parser.add_argument(
        "--checkpoint_model_name",
        type=str,
        default="protac-model",
        help="Name of the model checkpoint to save during training.",
    )
    parser.add_argument(
        "--finetuning_checkpoint",
        type=str,
        default=None,
        help="Path to a pre-trained model checkpoint for finetuning. If provided, the model will be loaded from this checkpoint and finetuned.",
    )
    main(parser.parse_args())