import pykoopman as pk
from pykoopman import observables
from EDMDSimulator import EDMDSimulator
import numpy as np

# 시스템 정의
def fwd_duffing(t, x, u=0, dt=0.1, delta=0.2, alpha=-1, beta=1):
    x1, x2 = x[0, :], x[1, :]
    dx1 = x1 + x2 * dt
    dx2 = x2 + (-delta * x2 - alpha * x1 - beta * x1**3) * dt
    return np.array([dx1, dx2])

# basis function 정의
RFF = observables.RandomFourierFeatures(
    include_state=True,
    gamma=1.0,
    D=100,
    random_state=42    # 시드
    )

centers = np.random.uniform(-2,2,(2,100))

RBF_thinplate = observables.RadialBasisFunction(
    rbf_type="thinplate",
    n_centers=centers.shape[1],
    centers=centers,
    kernel_width=1,
    polyharmonic_coeff=1.0,
    include_state=True,
)

RBF_gauss = observables.RadialBasisFunction(
    rbf_type="gauss",
    n_centers=centers.shape[1],
    kernel_width=1,
    include_state=True
)

RBF_invquad = observables.RadialBasisFunction(
    rbf_type="invquad",
    n_centers=centers.shape[1],
    kernel_width=1,
    include_state=True
)

RBF_invmultquad = observables.RadialBasisFunction(
    rbf_type="invmultquad",
    n_centers=centers.shape[1],
    kernel_width=1,
    include_state=True
)

RBF_polyharmonic = observables.RadialBasisFunction(
    rbf_type="polyharmonic",
    n_centers=centers.shape[1],
    polyharmonic_coeff=1.0,
    include_state=True
)

POLY = observables.Polynomial(
    degree=3,
    include_bias=True
)

sim = EDMDSimulator(
    system_name='duffing',
    system_function=fwd_duffing,
    n_states=2,
    dT=0.1,
    observable=RFF,
)

x0 = np.array([-0.5, -0.8])

sim.generate_training_data(xrange=(-2, 2), yrange=(-2, 2), grid_points=50)
sim.fit_model()
sim.evaluate_model(x0, step = 32)
sim.calculate_and_plot_training_error()
sim.plot_phase_portrait(x0)

# sim.plot_results(x0=np.array([-0.5, -0.8]))