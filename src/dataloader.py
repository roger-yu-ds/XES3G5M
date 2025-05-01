import numpy as np
import pandas as pd
import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from logging_config import get_logger

logger = get_logger(name=__name__)


class XES3G5MDataModuleConfig:
    """Configuration for the data module."""

    hf_dataset_ids: dict[str, str] = {
        "sequence": "Atomi/XES3G5M_interaction_sequences",
        "content_metadata": "Atomi/XES3G5M_content_metadata",
    }
    max_seq_length: int = 200
    padding_value: int = -1
    batch_size: int = 128
    val_fold: int = 4


class XES3G5MDataset(Dataset):
    """
    Dataset class for XES3G5M dataset.
    """

    def __init__(
        self,
        seq_df: pd.DataFrame,
        question_embeddings: np.ndarray,
        concept_embeddings: np.ndarray,
    ):
        """
        Initializes the dataset.
        """
        self.seq_df = seq_df
        self.question_embeddings = question_embeddings
        self.concept_embeddings = concept_embeddings

    def __len__(self) -> int:
        return len(self.seq_df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a dictionary of input tensors.
        """
        row = self.seq_df.iloc[idx]
        question_embeddings = self.question_embeddings[
            row["questions"]
        ]  # (padded_num_questions, emb_dim)
        concept_embeddings = self.concept_embeddings[
            row["concepts"]
        ]  # (padded_num_concepts, emb_dim)
        selectmasks = row["selectmasks"]
        responses = row["responses"]
        return {
            "questions": torch.LongTensor(row["questions"]),
            "concepts": torch.LongTensor(row["concepts"]),
            "question_embeddings": torch.Tensor(question_embeddings),
            "concept_embeddings": torch.Tensor(concept_embeddings),
            "selectmasks": torch.Tensor(selectmasks),
            "responses": torch.LongTensor(responses),
        }


class XES3G5MDataModule(pl.LightningDataModule):
    """
    DataModule class for XES3G5M dataset.
    """

    def __init__(
        self,
        config: XES3G5MDataModuleConfig,
    ) -> None:
        """
        Initializes the data module.
        """
        super().__init__()
        self.hf_dataset_ids = config.hf_dataset_ids
        self.batch_size = config.batch_size
        self.val_fold = config.val_fold
        self.max_seq_length = config.max_seq_length
        self.padding_value = config.padding_value

    def prepare_data(self) -> None:
        """
        Downloads the dataset.
        """
        [load_dataset(hf_dataset_id) for hf_dataset_id in self.hf_dataset_ids.values()]

    def _create_overlapping_rows_in_group(self, group: pd.DataFrame, overlap_size: int, max_seq_length: int = 200) -> pd.DataFrame:
        """
        Create overlapping rows for a given group of sequences.
        
        Args:
            group (pd.DataFrame): DataFrame containing sequences for a single user.
            overlap_size (int): Number of elements to overlap between consecutive sequences.
            max_seq_length (int): Maximum length of sequences (default: 200).
            
        Returns:
            pd.DataFrame: DataFrame with original and overlapping sequences.
        """
        # Sort the group by sequence index
        # group = group.sort_values('sequence_idx').reset_index(drop=True)
        seq_cols = ['questions', 'responses', 'selectmasks', 'concepts', "timestamps", "is_repeat"]
        # Initialize a list to store all rows (original + new overlapping ones)
        all_rows = []
        
        # Add original rows
        for _, row in group.iterrows():
            all_rows.append(row.to_dict())
        
        # Skip users with only one row (no need for overlap)
        if len(group) <= 1:
            return pd.DataFrame(all_rows)
        
        # For each pair of consecutive rows, create an overlapping row
        for i in range(len(group) - 1):
            current_row = group.iloc[i]
            next_row = group.iloc[i + 1]
            
            # Create new overlapping row
            overlap_row = {}
            
            # For sequence columns, take end of current and beginning of next
            for col in seq_cols:
                if col in current_row and col in next_row:
                    # Take last overlap_size elements from current row
                    first_part = current_row[col][-overlap_size:]
                    
                    # Take first (max_seq_length - overlap_size) elements from next row
                    second_part = next_row[col][:(max_seq_length - overlap_size)]
                    
                    # Combine them
                    overlap_row[col] = np.array(first_part.tolist() + second_part.tolist())
                    
                    # Pad if necessary (should rarely be needed if both sequences are max_seq_length)
                    if len(overlap_row[col]) < max_seq_length:
                        pad_value = -1 if col == 'selectmasks' else 0
                        padding = [pad_value] * (max_seq_length - len(overlap_row[col]))
                        overlap_row[col] = overlap_row[col] + padding
            
            # Copy other non-sequence columns from the first row
            for col in current_row.index:
                if col not in overlap_row and col not in seq_cols:
                    overlap_row[col] = current_row[col]
            
            # Add uid
            overlap_row['uid'] = current_row['uid']
            
            # Add the overlap row to the list
            all_rows.append(overlap_row)
        
        # Create new dataframe with all rows
        result_df = pd.DataFrame(all_rows)
        
        return result_df

    def create_overlapping_rows(
        self, df: pd.DataFrame, overlap_size: int, max_seq_length: int = 200
    ) -> pd.DataFrame:
        """Create overlapping rows for `df`. If rows with duplicate `uid`s would be extended; extra rows are created
        with overlapping sequences. This is done by taking the last `overlap_size` elements from the current row and the first `max_seq_length - overlap_size`
        elements from the next row.

        Args:
            df (pd.DataFrame): The DataFrame containing `uid` and the sequence columns.
            overlap_size (int): Number of elements to overlap between consecutive sequences.
            max_seq_length (int): Maximum length of sequences (default: 200).

        Returns:
            pd.DataFrame: DataFrame with original and overlapping sequences.
        """
        result = df.groupby("uid", as_index=False).apply(
            lambda group: self._create_overlapping_rows_in_group(
                group=group, overlap_size=overlap_size, max_seq_length=max_seq_length
            )
        ).reset_index(drop=True)
        return result
        

    def setup(
        self,
        stage: str | None = None,
        training_folds: list[int] = None,
        overlap_size: int = None,
    ) -> None:
        """
        Sets up the dataset for training, validation, and testing.

        Args:
            stage (str | None, optional): "fit" or "test". Defaults to None.
            training_folds (list[int], optional): The list of training folds to train the model on. Defaults to None.
            overlap_size (int, optional): The . Defaults to None.

        Raises:
            ValueError: _description_
        """
        datasets = {
            key: load_dataset(value) for key, value in self.hf_dataset_ids.items()
        }
        self.datasets = datasets
        if training_folds is None:
            training_folds = [0, 1, 2, 3]
        elif self.val_fold in training_folds:
            raise ValueError(
                f"Validation fold {self.val_fold} cannot be in training folds {training_folds}"
            )
        seq_df_train_val = datasets["sequence"]["train"].to_pandas()
        train_indices = seq_df_train_val["fold"].isin(training_folds)
        val_indices = seq_df_train_val["fold"] == self.val_fold
        self.seq_df_train = seq_df_train_val[train_indices]
        logger.info("self.seq_df_train.shape[0]: %s", self.seq_df_train.shape[0])
        if overlap_size:
            logger.info("Creating overlapping rows with overlap size %d", overlap_size)
            self.seq_df_train = self.create_overlapping_rows(
                self.seq_df_train, overlap_size=overlap_size
            )
        logger.info("self.seq_df_train.shape[0]: %s", self.seq_df_train.shape[0])
        self.seq_df_val = seq_df_train_val[val_indices]
        self.seq_df_test = datasets["sequence"]["test"].to_pandas()

        question_content_df = datasets["content_metadata"]["question"].to_pandas()
        concept_content_df = datasets["content_metadata"]["concept"].to_pandas()
        self.question_embeddings = np.array(
            [np.array(x) for x in question_content_df["embeddings"].values]
        )
        self.concept_embeddings = np.array(
            [np.array(x) for x in concept_content_df["embeddings"].values]
        )

        if stage == "fit" or stage is None:
            self.train_dataset = XES3G5MDataset(
                seq_df=self.seq_df_train,
                question_embeddings=self.question_embeddings,
                concept_embeddings=self.concept_embeddings,
            )
            self.val_dataset = XES3G5MDataset(
                seq_df=self.seq_df_val,
                question_embeddings=self.question_embeddings,
                concept_embeddings=self.concept_embeddings,
            )
        if stage == "test" or stage is None:
            self.test_dataset = XES3G5MDataset(
                seq_df=self.seq_df_test,
                question_embeddings=self.question_embeddings,
                concept_embeddings=self.concept_embeddings,
            )

    def _collate_fn(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """
        Collate function for the dataloader.
        """

        # Get list of tensors from the batch
        questions = [x["questions"] for x in batch]
        concepts = [x["concepts"] for x in batch]
        question_embeddings = [x["question_embeddings"] for x in batch]
        concept_embeddings = [x["concept_embeddings"] for x in batch]
        selectmasks = [x["selectmasks"] for x in batch]
        responses = [x["responses"] for x in batch]

        # Get the maximum sequence length in this batch
        max_len = max(x.shape[0] for x in questions)
        max_len = min(max_len, self.max_seq_length)  # Cap at max_seq_length if needed

        # Pad the sequences if not done already (for "test" mode)
        for i in range(len(questions)):
            seq_len = questions[i].shape[0]
            if seq_len < max_len:
                questions[i] = torch.nn.functional.pad(
                    questions[i], (0, max_len - seq_len), value=self.padding_value
                )
                concepts[i] = torch.nn.functional.pad(
                    concepts[i], (0, max_len - seq_len), value=self.padding_value
                )
                question_embeddings[i] = torch.nn.functional.pad(
                    question_embeddings[i], (0, 0, 0, max_len - seq_len), value=0
                )
                concept_embeddings[i] = torch.nn.functional.pad(
                    concept_embeddings[i], (0, 0, 0, max_len - seq_len), value=0
                )
                selectmasks[i] = torch.nn.functional.pad(
                    selectmasks[i], (0, max_len - seq_len), value=0
                )
                responses[i] = torch.nn.functional.pad(
                    responses[i], (0, max_len - seq_len), value=self.padding_value
                )
            else:
                questions[i] = questions[i][:max_len]
                concepts[i] = concepts[i][:max_len]
                question_embeddings[i] = question_embeddings[i][:max_len]
                concept_embeddings[i] = concept_embeddings[i][:max_len]
                selectmasks[i] = selectmasks[i][:max_len]
                responses[i] = responses[i][:max_len]

        # Stack the tensors
        stacked_questions = torch.stack(questions)  # (batch_size, max_seq_length)
        stacked_concepts = torch.stack(concepts)  # (batch_size, max_seq_length)
        stacked_question_embeddings = torch.stack(
            question_embeddings
        )  # (batch_size, max_seq_length, emb_dim)
        stacked_concept_embeddings = torch.stack(
            concept_embeddings
        )  # (batch_size, max_seq_length, emb_dim)
        stacked_selectmasks = torch.stack(selectmasks)  # (batch_size, max_seq_length)
        stacked_responses = torch.stack(responses)  # (batch_size, max_seq_length)

        # Replace padding value with 0 for responses
        stacked_responses[stacked_responses == self.padding_value] = 0

        return {
            "questions": stacked_questions,
            "concepts": stacked_concepts,
            "question_embeddings": stacked_question_embeddings,
            "concept_embeddings": stacked_concept_embeddings,
            "selectmasks": stacked_selectmasks,
            "responses": stacked_responses,
        }

    def train_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """
        Returns the training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """
        Returns the validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """
        Returns the test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
