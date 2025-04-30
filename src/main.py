from argparse import ArgumentParser
from pathlib import Path
import random
import numpy as np
import torch
import joblib
from datetime import datetime
from train import train_sakt_with_early_stopping
from models.sakt import SAKT, SAKTWithAdditivePreEmbeddings, SAKTConfig
from dataloader import XES3G5MDataModule, XES3G5MDataModuleConfig
from logging_config import get_logger

logger = get_logger(name=__name__)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Train SAKT model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/atomi/data_module.joblib",
        help="Path to the data module",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/sakt/",
        help="Directory to save the logs",
    )
    parser.add_argument(
        "--learning_curve",
        action="store_true",
        help="If set, the model will be trained in learning-curve mode",
    )
    parser.add_argument(
        "--pre_embedding_list",
        nargs="+",
        type=str,
        default=None,
        help="If set, the model will use question and/or concept pre-embeddings",
    )

    parser.add_argument(
        "--model", type=str, default="sakt", help="The model name to train"
    )
    parser.add_argument(
        "--run_name", type=str, default="sakt", help="The run name for Tensorboard"
    )
    return parser.parse_args()


def get_model(
    model_name: str, config: SAKTConfig, pre_embeddings: dict[str, torch.Tensor] | None
) -> torch.nn.Module:
    if model_name == "sakt":
        model = SAKT(
            emb_dim=config.emb_dim,
            max_seq_len=config.max_seq_len,
            num_questions=config.num_questions,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
    elif model_name == "sakt_with_additive_pre_embeddings":
        model = SAKTWithAdditivePreEmbeddings(
            emb_dim=config.emb_dim,
            max_seq_len=config.max_seq_len,
            num_questions=config.num_questions,
            num_heads=config.num_heads,
            pre_embeddings=pre_embeddings,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def main(args=None):
    """Main function to train the SAKT model."""
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_path = Path("data") / "atomi" / "data_module.joblib"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    if data_path.exists():
        logger.info("Loading data locally from %s", data_path)
        data_module = joblib.load(data_path)
    else:
        data_module = XES3G5MDataModule(config=XES3G5MDataModuleConfig())
        logger.info("Preparing data...")
        data_module.prepare_data()
        logger.info("Setting up data...")
        data_module.setup(stage="fit")
        data_module.setup(stage="test")
        joblib.dump(data_module, data_path)

    if args.learning_curve:
        logger.info("Using learning-curve mode")
        training_folds_list = [list(range(i + 1)) for i in range(4)]
        logger.info("Training on folds: %s", training_folds_list)
    else:
        training_folds_list = [list(range(4))]

    # Add loop to train on different training set sizes
    for training_folds in training_folds_list:
        folds_name = "_".join(str(f) for f in training_folds)
        log_dir = Path("logs") / "sakt" / f"{args.model}_{folds_name}_{now_str}"
        log_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = Path("artifacts") / "sakt" / f"{folds_name}_{now_str}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        data_module.setup(stage="fit", training_folds=training_folds)
        logger.info("Training on folds: %s", training_folds)

        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
        logger.info("Train dataloader and val dataloader created.")
        # Passing the pre-embeddings (if any) to the model
        pre_embeddings = {}
        for emb_name in args.pre_embedding_list or []:
            try:
                embeddings = getattr(data_module, f"{emb_name}_embeddings")
            except AttributeError:
                logger.error(
                    f"Pre-embedding {emb_name} not found in data module. Available: `question`, `concept`"
                )
                continue
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
            pre_embeddings[emb_name] = embeddings
            logger.info(
                f"Pre-embedding {emb_name} loaded with shape: {pre_embeddings[emb_name].shape}"
            )

        model = get_model(
            model_name=args.model,
            config=SAKTConfig(),
            pre_embeddings=pre_embeddings,
        )
        logger.info("Model initialized.")
        logger.info("Training model...")
        logger.info("Training on %d samples", len(train_dataloader.dataset))
        model, best_val_loss, best_epoch = train_sakt_with_early_stopping(
            model=model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            device=device,
            log_dir=log_dir,
            epochs=100,
            learning_rate=0.001,
            patience=10,
        )
        logger.info("Training completed.")
        model_path = artifact_dir / f"{args.model}_best_model.pth"
        torch.save(model.state_dict(), model_path)
        logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
