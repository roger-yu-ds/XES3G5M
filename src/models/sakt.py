import torch
from torch.nn import (
    Module,
    Embedding,
    Linear,
    MultiheadAttention,
    LayerNorm,
    Dropout,
    Sequential,
    ReLU,
    Sigmoid,
)
from logging_config import get_logger

logger = get_logger(name=__name__)

device = "cpu" if not torch.cuda.is_available() else "cuda"


class SAKTConfig:
    """Configuration for the SAKT class."""

    emb_dim: int = 768
    max_seq_len: int = 200
    num_questions: int = 7652
    num_concepts: int = 865
    num_heads: int = 6
    dropout: float = 0.2


class FFN(Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
            Linear(self.emb_size, self.emb_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.emb_size, self.emb_size),
        )

    def forward(self, inputs):
        return self.FFN(inputs)


class SAKT(Module):
    def __init__(
        self,
        emb_dim: int,
        max_seq_len: int,
        num_questions: int,
        num_heads: int,
        dropout: float = 0.2,
    ) -> None:
        """SAKT model for knowledge tracing.

        Args:
            emb_dim (int): The embedding dimension.
            max_seq_len (int): The maximum sequence length.
            num_questions (int): The number of questions.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout rate. Defaults to 0.2.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.num_questions = num_questions
        self.num_heads = num_heads
        self.dropout = dropout
        # Use the last row of the embedding matrix for padding.
        self.question_padding_idx = num_questions
        self.interaction_padding_idx = num_questions * 2
        self.position_padding_idx = max_seq_len

        # Embedding layers
        # Add a row reserved for padding, i.e. values of -1 because there is no actual event.
        self.question_embedding = Embedding(
            num_embeddings=self.num_questions + 1,
            embedding_dim=self.emb_dim,
            padding_idx=self.question_padding_idx,
        )
        # The interaction of the question_id and whether the answer is correct (binary), hence times 2.
        self.interaction_embedding = Embedding(
            num_embeddings=self.num_questions * 2 + 1,
            embedding_dim=self.emb_dim,
            padding_idx=self.interaction_padding_idx,
        )
        self.position_embedding = Embedding(
            num_embeddings=self.max_seq_len + 1,
            embedding_dim=self.emb_dim,
            padding_idx=self.position_padding_idx,
        )

        self.attention = MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=self.dropout,
        )
        self.register_buffer(
            "attn_mask",
            torch.triu(
                torch.ones(self.max_seq_len, self.max_seq_len), diagonal=1
            ).bool(),
        )

        # Feedforward network
        self.ffn = FFN(self.emb_dim, self.dropout)

        # Layer normalization
        self.layer_norm1 = LayerNorm(self.emb_dim)
        self.layer_norm2 = LayerNorm(self.emb_dim)

        # Output layer
        self.output_layer = Linear(self.emb_dim, 1)

    def forward(
        self,
        question_ids: torch.Tensor,
        responses: torch.Tensor,
        selectmasks: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the SAKT model.

        Args:
            question_ids (torch.Tensor): The question IDs (batch_size, seq_len).
            responses (torch.Tensor): The responses (batch_size, seq_len).
            selectmasks (torch.Tensor): The selection masks (batch_size, seq_len), 1 for
                selected questions, -1 for padding.

        Returns:
            torch.Tensor: The output of the model.
        """
        seq_len = question_ids.size(1)
        batch_size = question_ids.size(0)
        # Embedding lookup, replace with the respective padding indexex.
        question_ids = torch.where(
            condition=selectmasks == -1,
            input=torch.full_like(question_ids, self.question_padding_idx),
            other=question_ids,
        )
        question_embedding = self.question_embedding(
            question_ids
        )  # (batch_size, seq_len, emb_dim)
        interaction_ids = question_ids + self.num_questions * responses
        interaction_ids = torch.where(
            condition=selectmasks == -1,
            input=interaction_ids,
            other=torch.full_like(interaction_ids, self.interaction_padding_idx),
        )
        interaction_embedding = self.interaction_embedding(interaction_ids)

        position_ids = (
            torch.arange(seq_len, device=question_ids.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        position_ids = torch.where(
            condition=selectmasks == -1,
            input=torch.full_like(
                position_ids, self.position_padding_idx, device=question_ids.device
            ),
            other=position_ids,
        )
        position_embedding = self.position_embedding(position_ids)

        # Combine embeddings
        x = interaction_embedding + position_embedding

        # Attention layer
        key_padding_mask = selectmasks == -1
        x, _ = self.attention(
            query=question_embedding,
            key=x,
            value=x,
            attn_mask=self.attn_mask,
            key_padding_mask=key_padding_mask,
        )

        # Skip connection
        x = self.layer_norm1(x + question_embedding)

        # Feedforward network
        ffn_out = self.ffn(x)

        # Skip connection
        x = self.layer_norm2(x + ffn_out)

        # Output layer
        logits = self.output_layer(x)
        return logits


class SAKTWithAdditivePreEmbeddings(Module):
    def __init__(
        self,
        emb_dim: int,
        max_seq_len: int,
        num_questions: int,
        num_heads: int,
        pre_embeddings: dict[str, torch.Tensor],
        dropout: float = 0.2,
    ) -> None:
        """SAKT model for knowledge tracing.

        Args:
            emb_dim (int): The embedding dimension.
            max_seq_len (int): The maximum sequence length.
            num_questions (int): The number of questions.
            num_heads (int): The number of attention heads.
            pre_embeddings (dict[str, torch.Tensor]): A dictionary of pre-embeddings to be added to the learned embeddings; must be the same dimension as the `emb_dim`.
            dropout (float, optional): The dropout rate. Defaults to 0.2.
        """
        super().__init__()
        # Register pre-embeddings as buffers so they are not considered parameters.
        self.pre_embedding_names = []
        if pre_embeddings:
            for emb_name, pre_embedding_map in pre_embeddings.items():
                if pre_embedding_map.shape[1] != emb_dim:
                    logger.warning(
                        f"Pre-embedding {emb_name} has a different dimension than the embedding dimension. Skipping."
                    )
                    continue
                # Add a row reserved for padding, i.e. values of -1 because there is no actual event.
                pre_embedding_map = torch.cat(
                    [pre_embedding_map, torch.zeros(1, emb_dim)], dim=0
                )
                pre_embedding_name = f"{emb_name}_pre_embedding_map"
                logger.info(
                    "Registering pre-embedding %s as %s, with shape %s",
                    emb_name,
                    pre_embedding_name,
                    pre_embedding_map.shape,
                )
                self.register_buffer(pre_embedding_name, pre_embedding_map)

                self.pre_embedding_names.append(pre_embedding_name)

        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.num_questions = num_questions
        self.num_heads = num_heads
        self.pre_embeddings = pre_embeddings
        self.dropout = dropout
        # Use the last row of the embedding matrix for padding.
        self.question_padding_idx = num_questions
        self.interaction_padding_idx = num_questions * 2
        self.position_padding_idx = max_seq_len

        # Embedding layers
        # Add a row reserved for padding, i.e. values of -1 because there is no actual event.
        self.question_embedding = Embedding(
            num_embeddings=self.num_questions + 1,
            embedding_dim=self.emb_dim,
            padding_idx=self.question_padding_idx,
        )
        # The interaction of the question_id and whether the answer is correct (binary), hence times 2.
        self.interaction_embedding = Embedding(
            num_embeddings=self.num_questions * 2 + 1,
            embedding_dim=self.emb_dim,
            padding_idx=self.interaction_padding_idx,
        )
        self.position_embedding = Embedding(
            num_embeddings=self.max_seq_len + 1,
            embedding_dim=self.emb_dim,
            padding_idx=self.position_padding_idx,
        )

        self.attention = MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=self.dropout,
        )
        self.register_buffer(
            "attn_mask",
            torch.triu(
                torch.ones(self.max_seq_len, self.max_seq_len), diagonal=1
            ).bool(),
        )

        # Feedforward network
        self.ffn = FFN(self.emb_dim, self.dropout)

        # Layer normalization
        self.layer_norm1 = LayerNorm(self.emb_dim)
        self.layer_norm2 = LayerNorm(self.emb_dim)

        # Output layer
        self.output_layer = Linear(self.emb_dim, 1)

    def forward(
        self,
        question_ids: torch.Tensor,
        responses: torch.Tensor,
        selectmasks: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the SAKT model.

        Args:
            question_ids (torch.Tensor): The question IDs (batch_size, seq_len).
            responses (torch.Tensor): The responses (batch_size, seq_len).
            selectmasks (torch.Tensor): The selection masks (batch_size, seq_len), 1 for
                selected questions, -1 for padding.

        Returns:
            torch.Tensor: The output of the model.
        """
        seq_len = question_ids.size(1)
        batch_size = question_ids.size(0)
        # Embedding lookup, replace with the respective padding indexex.
        question_ids = torch.where(
            condition=selectmasks == -1,
            input=torch.full_like(question_ids, self.question_padding_idx),
            other=question_ids,
        )
        question_embedding = self.question_embedding(
            question_ids
        )  # (batch_size, seq_len, emb_dim)

        interaction_ids = question_ids + self.num_questions * responses
        interaction_ids = torch.where(
            condition=selectmasks == -1,
            input=interaction_ids,
            other=torch.full_like(interaction_ids, self.interaction_padding_idx),
        )
        interaction_embedding = self.interaction_embedding(interaction_ids)

        position_ids = (
            torch.arange(seq_len, device=question_ids.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        position_ids = torch.where(
            condition=selectmasks == -1,
            input=torch.full_like(
                position_ids, self.position_padding_idx, device=question_ids.device
            ),
            other=position_ids,
        )
        position_embedding = self.position_embedding(position_ids)

        # Combine embeddings
        x = interaction_embedding + position_embedding
        if self.pre_embedding_names:
            for pre_embedding_name in self.pre_embedding_names:
                pre_embedding = getattr(self, pre_embedding_name)
                # Select the pre-embedding based on the question IDs
                pre_embedding = pre_embedding[question_ids]
                x += pre_embedding

        # Attention layer
        key_padding_mask = selectmasks == -1
        x, _ = self.attention(
            query=question_embedding,
            key=x,
            value=x,
            attn_mask=self.attn_mask,
            key_padding_mask=key_padding_mask,
        )

        # Skip connection
        x = self.layer_norm1(x + question_embedding)

        # Feedforward network
        ffn_out = self.ffn(x)

        # Skip connection
        x = self.layer_norm2(x + ffn_out)

        # Output layer
        logits = self.output_layer(x)
        return logits
