import sys
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from logging_config import get_logger

logger = get_logger(name=__name__)


def mask_loss(
    loss: torch.Tensor,
    selectmask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask the loss based on the selectmask.

    Args:
        loss (torch.Tensor): The loss tensor.
        selectmask (torch.Tensor): The selectmask tensor.

    Returns:
        tuple(torch.Tensor, torch.Tensor, torch.Tensor): The mean masked loss, total masked loss and the count of valid samples.
    """
    # Mask the loss
    loss_mask = selectmask != -1
    masked_loss = loss * loss_mask.float()
    return masked_loss.sum() / loss_mask.sum(), masked_loss.sum(), loss_mask.sum()


def train_sakt_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    log_dir: Path,
    epochs: int = 20,
    learning_rate: float = 0.001,
    patience: int = 3,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,

):
    """Train the model with early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): The training data loader.
        device (str): The device to use for training, e.g., "cuda" or "cpu".
        log_dir (Path): The directory to save the Tensorboard logs.
        epochs (int, optional): The number of epochs to train. Defaults to 20.
        learning_rate (float, optional): The learning rate. Defaults to 0.001.
        patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 3.
        val_loader (DataLoader, optional): The validation data loader. Defaults to None.
        test_loader (DataLoader, optional): The test data loader. Defaults to None.
    """
    train_size = len(train_loader.dataset)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    best_model_state = None
    # At the epoch with the best validation loss; for learning curve.
    corresponding_train_loss = float("inf")
    corresponding_train_acc = 0
    corresponding_train_auc = 0

    model.to(device)

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        all_train_probs = []
        all_train_labels = []
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", file=sys.stdout
        )
        for batch in progress_bar:
            questions = batch["questions"].to(device)
            selectmasks = batch["selectmasks"].to(device)
            responses = batch["responses"].to(device)

            logits = model(
                questions,
                responses,
                selectmasks,
            )
            loss = criterion(logits.squeeze(-1), responses.float())
            # Mask the loss
            mean_loss, batch_loss, batch_count = mask_loss(loss, selectmasks[:, 1:])
            optimizer.zero_grad()
            mean_loss.backward()

            optimizer.step()

            running_loss += batch_loss.item()
            running_count += batch_count.item()

            # Calculate accuracy and AUC
            probs = torch.sigmoid(logits)
            mask = selectmasks != -1
            all_train_probs.append(probs[mask].detach().cpu())
            all_train_labels.append(responses[mask].detach().cpu())

        epoch_loss = running_loss / running_count

        # Compute train accuracy and AUC
        train_probs = torch.cat(all_train_probs).numpy()
        train_labels = torch.cat(all_train_labels).numpy()
        train_preds = (train_probs >= 0.5).astype(int)
        train_acc = accuracy_score(train_labels, train_preds)
        try:
            train_auc = roc_auc_score(train_labels, train_probs)
        except ValueError:
            train_auc = float("nan")
        
        # TensorBoard logging
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("AUC/Train", train_auc, epoch)

        # Validation
        if val_loader:
            model.eval()
            running_val_loss = 0.0
            running_val_count = 0.0
            all_val_probs = []
            all_val_labels = []
            progress_bar = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs}", file=sys.stdout
            )
            logger.info("Validating...")
            with torch.no_grad():
                for batch in progress_bar:
                    questions = batch["questions"].to(device)
                    selectmasks = batch["selectmasks"].to(device)
                    responses = batch["responses"].to(device)

                    logits = model(
                        questions,
                        responses,
                        selectmasks,
                    )
                    _, batch_val_loss, batch_val_count = mask_loss(
                    running_val_loss += batch_val_loss.item()
                    running_val_count += batch_val_count.item()

                    # Calculate accuracy and AUC
                    val_probs = torch.sigmoid(logits)
                    mask = selectmasks != -1
                    all_val_probs.append(val_probs[mask].detach().cpu())
                    all_val_labels.append(responses[mask].detach().cpu())

            epoch_val_loss = running_val_loss / running_val_count
            # Compute validation accuracy and AUC
            val_probs = torch.cat(all_val_probs).numpy()
            val_labels = torch.cat(all_val_labels).numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            val_acc = accuracy_score(val_labels, val_preds)
            try:
                val_auc = roc_auc_score(val_labels, val_probs)
            except ValueError:
                logger.warning("AUC calculation failed due to no positive samples.")
                val_auc = float("nan")
                # TensorBoard logging
            writer.add_scalar("Loss/Val", epoch_val_loss, epoch)
            writer.add_scalar("Accuracy/Val", val_acc, epoch)
            writer.add_scalar("AUC/Val", val_auc, epoch)


        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
        )
        logger.info(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
        logger.info(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_loader:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                no_improvement = 0
                best_model_state = model.state_dict()
                corresponding_train_loss = epoch_loss
                corresponding_train_acc = train_acc
                corresponding_train_auc = train_auc
            else:
                no_improvement += 1

            if no_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break
    
            writer.add_scalar("Best Loss/Train", corresponding_train_loss, train_size)
            writer.add_scalar("Best Loss/Val", best_val_loss, train_size)
            writer.add_scalar("Best Accuracy/Train", corresponding_train_acc, train_size)
            writer.add_scalar("Best Accuracy/Val", val_acc, train_size)
            writer.add_scalar("Best AUC/Train", corresponding_train_auc, train_size)
            writer.add_scalar("Best AUC/Val", val_auc, train_size)
            epoch_loss = best_val_loss
            epoch = best_epoch
            probs = val_probs

            # Load the best model state
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                logger.info(f"Loaded best model state from epoch {best_epoch}")
            else:
                logger.warning("No best model state found. Training may not have improved.")
    writer.close()
    return model, epoch_loss, epoch, probs


def evaluate_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
):
    """Evaluate the model on the test set.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): The test data loader.
        device (str): The device to use for evaluation, e.g., "cuda" or "cpu".
    """
    model.eval()
    all_test_probs = []
    all_test_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", file=sys.stdout):
            questions = batch["questions"].to(device)
            selectmasks = batch["selectmasks"].to(device)
            responses = batch["responses"].to(device)

            logits = model(
                questions,
                responses,
                selectmasks,
            )

            probs = torch.sigmoid(logits)
            mask = selectmasks != -1
            all_test_probs.append(probs[mask].detach().cpu())
            all_test_labels.append(responses[mask].detach().cpu())

    test_probs = torch.cat(all_test_probs).numpy()
    test_labels = torch.cat(all_test_labels).numpy()
    test_preds = (test_probs >= 0.5).astype(int)
    test_acc = accuracy_score(test_labels, test_preds)
    try:
        test_auc = roc_auc_score(test_labels, test_probs)
    except ValueError:
        logger.warning("AUC calculation failed due to no positive samples.")
        test_auc = float("nan")

    logger.info(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    return test_acc, test_auc