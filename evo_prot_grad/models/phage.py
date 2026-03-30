import torch
import torch.nn as nn


class ValidStaple(nn.Module):
    def __init__(self, staple_token: int):
        """_summary_

        Args:
            staple_token (int): _description_

        Example:
        # Construct X: batch=8, seq_len=16, vocab_size=34, integer tokens
        batch_size = 8
        seq_len = 16
        vocab_size = 34
        staple_token = 33

        # Start with random tokens in [0, vocab_size-2] (exclude 33 for now)
        X = torch.randint(0, vocab_size - 1, (batch_size, seq_len))

        # For each sequence, insert at least 2 staple tokens (token 33), some with more
        for i in range(batch_size):
            # Randomly choose 2-4 positions to set as staple_token
            n_staples = torch.randint(2, 5, (1,)).item()
            staple_positions = torch.randperm(seq_len)[:n_staples]
            X[i, staple_positions] = staple_token
        model = ValidStaple(staple_token=33)
        model(X)
        """
        super().__init__()
        self.staple_token = staple_token

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: batch x seq_len x vocab size
        n_staple_toks = X[:, :, self.staple_token].sum(axis=1)
        return -1e6 * torch.relu(n_staple_toks - 2)
