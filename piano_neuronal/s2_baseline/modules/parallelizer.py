"""Parallelizer for distributing polyphonic notes across synth slots.

Merges batch and polyphony axes for efficient monophonic processing,
then unmerges to distribute outputs back to voice slots.
"""

import torch


class Parallelizer:
    """Utility for (un)merging batch and polyphony dimensions.

    The DDSP-Piano model processes each voice independently through the
    MonophonicNetwork. The Parallelizer reshapes the batch so that all
    voices across all batch items are processed in parallel.

    This is a stateless utility (not an nn.Module).
    """

    @staticmethod
    def parallelize(
        *tensors: torch.Tensor,
        n_synths: int,
    ) -> tuple:
        """Merge batch and polyphony axes.

        Args:
            tensors: each of shape (B, T, n_synths, D) or (B, T, D).
            n_synths: number of synth slots.

        Returns:
            tuple of tensors with shape (B*n_synths, T, D) or (B*n_synths, T).
        """
        results = []
        for t in tensors:
            if t.dim() == 4 and t.shape[2] == n_synths:
                # (B, T, n_synths, D) -> (B*n_synths, T, D)
                B, T, S, D = t.shape
                results.append(t.reshape(B * S, T, D))
            elif t.dim() == 3 and t.shape[2] == n_synths:
                # (B, T, n_synths) -> (B*n_synths, T)
                B, T, S = t.shape
                results.append(t.reshape(B * S, T))
            elif t.dim() == 2:
                # (B, D) -> (B*n_synths, D)
                B, D = t.shape
                results.append(t.unsqueeze(1).expand(-1, n_synths, -1).reshape(B * n_synths, D))
            else:
                # Already parallel or unknown shape — pass through
                results.append(t)
        return tuple(results)

    @staticmethod
    def unparallelize(
        *tensors: torch.Tensor,
        batch_size: int,
        n_synths: int,
    ) -> tuple:
        """Unmerge batch and polyphony axes.

        Args:
            tensors: each of shape (B*n_synths, T, D) or (B*n_synths, T).
            batch_size: original batch size B.
            n_synths: number of synth slots.

        Returns:
            tuple of tensors with shape (B, T, n_synths, D) or (B, T, n_synths).
        """
        results = []
        for t in tensors:
            if t.dim() == 3:
                # (B*n_synths, T, D) -> (B, T, n_synths, D)
                results.append(t.reshape(batch_size, n_synths, t.shape[1], t.shape[2])
                               .permute(0, 2, 1, 3))
            elif t.dim() == 2:
                # (B*n_synths, T) -> (B, T, n_synths)
                results.append(t.reshape(batch_size, n_synths, t.shape[1])
                               .permute(0, 2, 1))
            else:
                results.append(t)
        return tuple(results)