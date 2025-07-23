import pykoopman as pk
from pykoopman import observables
from EDMDSimulator import EDMDSimulator
import numpy as np

# 시스템 정의
def fwd_lotka_volterra(t, x, u=0, dt=0.1, alpha=1.5, beta=1.0, gamma=3.0, delta=1.0):
    x1, x2 = x[0, :], x[1, :]
    dx1 = x1 + dt * (alpha * x1 - beta * x1 * x2)
    dx2 = x2 + dt * (delta * x1 * x2 - gamma * x2)
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
    system_name='lotka_volterra',
    system_function=fwd_lotka_volterra,
    n_states=2,
    dT=0.1,
    n_traj=50,
    n_int=8,
    futuresize=1.8,
    observable=RFF,
)

sim.generate_training_data()
sim.fit_model()
sim.evaluate_model(x0=np.array([-0.5, -0.8]))
sim.predict_on_training()
sim.plot_results(x0=np.array([-0.5, -0.8]))