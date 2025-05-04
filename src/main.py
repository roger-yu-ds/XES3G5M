from argparse import ArgumentParser
from pathlib import Path
import random
import numpy as np
import torch
import joblib
from datetime import datetime
from train import evaluate_test, train_sakt_with_early_stopping
from models.sakt import SAKT, SAKTConfig
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
        "--overlap_size",
        type=int,
        default=None,
        help="If set, the training set will be extended to include overlapping sequences",
    )
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        help="If set, the model will evaluate the test set using {model_dir}_best_model.pth",
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
    model_name: str,
    config: SAKTConfig,
    pre_embeddings: dict[str, torch.Tensor] | None,
    pretrained_model_dir: str | None = None,
) -> torch.nn.Module:
    """Get the model based on the model name. Or load a pretrained model from the model_dir.

    Args:
        model_name (str): The name of the model to train.
        config (SAKTConfig): The configuration for the SAKT model.
        pre_embeddings (dict[str, torch.Tensor] | None): The pre-embeddings for the model. Defaults to None.
        model_dir (str | None, optional): The directory containing `best_model.pth`. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        torch.nn.Module: _description_
    """

    model = SAKT(
        emb_dim=config.emb_dim,
        max_seq_len=config.max_seq_len,
        num_questions=config.num_questions,
        num_heads=config.num_heads,
        dropout=config.dropout,
        pre_embeddings=pre_embeddings,
    )

    if pretrained_model_dir:
        logger.info("Loading model from %s", pretrained_model_dir)
        model_path = Path(pretrained_model_dir) / "best_model.pth"
        if not model_path.exists():
            logger.error("Model path does not exist: %s", model_path)
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        model.load_state_dict(torch.load(model_path))
        logger.info("Model loaded successfully")
    return model


def get_pre_embeddings(
    pre_embedding_list: list[str],
    data_module: XES3G5MDataModule,
) -> dict[str, torch.Tensor] | None:
    """Get the pre-embeddings for the model.

    Args:
        pre_embedding_list (list[str] | None): The list of pre-embeddings to use.
        data_module (XES3G5MDataModule): The data module.

    Returns:
        dict[str, torch.Tensor] | None: The pre-embeddings for the model.
    """
    if pre_embedding_list is None:
        return None
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
    return pre_embeddings


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
        joblib.dump(data_module, data_path)

    pre_embeddings = get_pre_embeddings(
        pre_embedding_list=args.pre_embedding_list,
        data_module=data_module,
    )

    # Evaluate the test set
    pretrained_model_dir = args.pretrained_model_dir
    if pretrained_model_dir:
        logger.info("Evaluating test set using model from %s", pretrained_model_dir)
        model_path = Path(pretrained_model_dir) / "best_model.pth"
        if not model_path.exists():
            logger.error("Model path does not exist: %s", model_path)
            return
        model = get_model(
            model_name=args.model,
            config=SAKTConfig(),
            pre_embeddings=pre_embeddings,
            pretrained_model_dir=args.pretrained_model_dir,
        )

        model.to(device)
        data_module.setup(
            stage="test",
        )
        test_dataloader = data_module.test_dataloader()
        logger.info("Evaluating test set...")
        evaluate_test(model, test_dataloader, device)
        return

    if args.learning_curve:
        logger.info("Using learning-curve mode")
        training_folds_list = [list(range(i + 1)) for i in range(4)]
        logger.info("Training on folds: %s", training_folds_list)
    else:
        training_folds_list = [list(range(4))]

    # Add loop to train on different training set sizes
    for training_folds in training_folds_list:
        folds_name = "_".join(str(f) for f in training_folds)
        log_dir = Path("logs") / "sakt" / f"{args.run_name}_{folds_name}_{now_str}"
        log_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = (
            Path("artifacts") / "sakt" / f"{args.run_name}_{folds_name}_{now_str}"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Setting up data: stage: %s, training_folds: %s, overlap_size: %s",
            "fit",
            training_folds,
            args.overlap_size,
        )
        data_module.setup(
            stage="fit", training_folds=training_folds, overlap_size=args.overlap_size
        )
        logger.info("Training on folds: %s", training_folds)

        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
        logger.info("Train dataloader and val dataloader created.")
        # Passing the pre-embeddings (if any) to the model

        model = get_model(
            model_name=args.model,
            config=SAKTConfig(),
            pre_embeddings=pre_embeddings,
        )
        logger.info("Model initialized.")
        logger.info("Training model...")
        logger.info("Training on %d samples", len(train_dataloader.dataset))
        model, best_val_loss, best_epoch, probs = train_sakt_with_early_stopping(
            model=model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            device=device,
            log_dir=log_dir,
            epochs=100,
            learning_rate=0.001,
            patience=10,
            pre_embedding_names=args.pre_embedding_list,
        )
        logger.info("Training completed.")
        model_path = artifact_dir / f"best_model.pth"
        torch.save(model.state_dict(), model_path)
        logger.info("Model saved to %s", model_path)
        probs_path = artifact_dir / f"probs.joblib"
        joblib.dump(probs, probs_path)
        logger.info("Probabilities saved to %s", probs_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
