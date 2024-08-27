import torch
import numpy as np


class torch_default_dtype:

    def __init__(self, dtype):
        self.saved_dtype = None
        self.dtype = dtype

    def __enter__(self):
        self.saved_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_default_dtype(self.saved_dtype)


def irr_repr(order, alpha, beta, gamma, dtype=None):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    # from from_lielearn_SO3.wigner_d import wigner_D_matrix
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
    # if order == 1:
    #     # change of basis to have vector_field[x, y, z] = [vx, vy, vz]
    #     A = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    #     return A @ wigner_D_matrix(1, alpha, beta, gamma) @ A.T

    # TODO (non-essential): try to do everything in torch
    # return torch.tensor(wigner_D_matrix(torch.tensor(order), alpha, beta, gamma), dtype=torch.get_default_dtype() if dtype is None else dtype)
    return torch.tensor(wigner_D_matrix(order, np.array(alpha), np.array(beta), np.array(gamma)), dtype=torch.get_default_dtype() if dtype is None else dtype)