import numpy as np
import torch

def torch_lu(A):
    """
    Perform LU decomposition with partial pivoting.

    Parameters
    ----------
        A (torch.Tensor): The input square matrix (n x n).

    Returns
    -------
        L (torch.Tensor): Lower triangular matrix with ones on the diagonal.
        U (torch.Tensor): Upper triangular matrix.
        P (torch.Tensor): Permutation matrix.
    """
    n = A.shape[0]
    A = A.clone()  # Work on a copy of A to avoid modifying the original
    P = torch.eye(n, device=A.device, dtype=A.dtype)  # Permutation matrix
    L = torch.zeros_like(A)  # Lower triangular matrix
    U = torch.zeros_like(A)  # Upper triangular matrix

    for i in range(n):
        # Pivoting: Find the row with the maximum element in the current column
        pivot = torch.argmax(torch.abs(A[i:, i])) + i

        # Swap rows in A and P for pivoting
        if pivot != i:
            A[[i, pivot]] = A[[pivot, i]]
            P[[i, pivot]] = P[[pivot, i]]

        # Fill the L matrix below the diagonal
        L[i, i] = 1
        for j in range(i + 1, n):
            L[j, i] = A[j, i] / A[i, i]
            A[j, i:] -= L[j, i] * A[i, i:]

        # Fill the U matrix for the current row
        U[i, i:] = A[i, i:]

    return L, U, P

def torch_solve(A, b):
    """
    Solve the linear system Ax = b using LU decomposition.

    Parameters
    ----------
        A (torch.Tensor): Coefficient matrix (n x n).
        b (torch.Tensor): Right-hand side matrix/vector (n x m).

    Returns
    -------
        torch.Tensor: Solution to the linear system (n x m).
    """
    # Perform LU decomposition: A = P L U
    L, U, P = torch_lu(A)

    # Solve Ly = P @ b for y (forward substitution)
    y = torch.triangular_solve(P @ b, L, upper=False, unitriangular=True).solution

    # Solve Ux = y for x (backward substitution)
    x = torch.triangular_solve(y, U, upper=True).solution

    return x

def torch_lstsq(A, b):
    """
    Solve the least-squares problem Ax = b using LU decomposition.

    Parameters
    ----------
        A (torch.Tensor): Design matrix (n x m) with n samples and m features.
        b (torch.Tensor): Observation vector (n x 1).

    Returns
    -------
        torch.Tensor: Solution vector (m x 1) that minimizes ||Ax - b||^2.
    """
    # Compute A^T A and A^T b
    AtA = torch.matmul(A.T, A)
    Atb = torch.matmul(A.T, b)

    # Solve the normal equations AtA x = Atb using torch_solve
    x = torch_solve(AtA, Atb)

    return x