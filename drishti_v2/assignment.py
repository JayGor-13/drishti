from __future__ import annotations

import torch
from torch import Tensor


def linear_sum_assignment(cost: Tensor) -> tuple[Tensor, Tensor]:
    """Return the minimum-cost one-to-one assignment for a 2-D cost tensor.

    This is a small rectangular Hungarian implementation. Keeping it local
    avoids making inference depend on SciPy while still providing globally
    optimal matching for crop assignment, temporal alignment, and tracking.
    The discrete assignment is intentionally computed outside autograd.
    """

    if cost.ndim != 2:
        raise ValueError(f"Expected a 2-D cost matrix, got {tuple(cost.shape)}")
    rows, cols = cost.shape
    device = cost.device
    if rows == 0 or cols == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    transposed = rows > cols
    work = cost.detach().to(device="cpu", dtype=torch.float64)
    if transposed:
        work = work.transpose(0, 1)
    num_rows, num_cols = work.shape
    # Read the matrix into plain Python lists once. Indexing a torch tensor element by element
    # inside the loops below costs a full tensor op per read (~160k per forward pass), which made
    # training ~15x slower on Windows. Same numbers, same result, pure Python floats.
    work = work.tolist()

    # Hungarian shortest augmenting-path algorithm for num_rows <= num_cols.
    u = [0.0] * (num_rows + 1)
    v = [0.0] * (num_cols + 1)
    matched_row = [0] * (num_cols + 1)
    previous_col = [0] * (num_cols + 1)

    for row in range(1, num_rows + 1):
        matched_row[0] = row
        min_value = [float("inf")] * (num_cols + 1)
        used = [False] * (num_cols + 1)
        col = 0
        while True:
            used[col] = True
            current_row = matched_row[col]
            delta = float("inf")
            next_col = 0
            for candidate in range(1, num_cols + 1):
                if used[candidate]:
                    continue
                reduced = work[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                if reduced < min_value[candidate]:
                    min_value[candidate] = reduced
                    previous_col[candidate] = col
                if min_value[candidate] < delta:
                    delta = min_value[candidate]
                    next_col = candidate
            for candidate in range(num_cols + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    min_value[candidate] -= delta
            col = next_col
            if matched_row[col] == 0:
                break

        while True:
            prior = previous_col[col]
            matched_row[col] = matched_row[prior]
            col = prior
            if col == 0:
                break

    row_indices = []
    col_indices = []
    for col in range(1, num_cols + 1):
        if matched_row[col] != 0:
            row_indices.append(matched_row[col] - 1)
            col_indices.append(col - 1)

    row_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)
    col_tensor = torch.tensor(col_indices, dtype=torch.long, device=device)
    if transposed:
        return col_tensor, row_tensor
    return row_tensor, col_tensor
