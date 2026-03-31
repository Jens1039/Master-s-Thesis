import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from firedrake import *
from firedrake.adjoint import *

from nondimensionalization import second_nondimensionalisation
from perturbed_flow_return_UFL import perturbed_flow_differentiable, NumpyLinSolveBlock, numpy_lin_solve_to_R, stokes_solve

if __name__ == "__main__":
    pass



