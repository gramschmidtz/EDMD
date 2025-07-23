import pykoopman as pk
from pykoopman import observables
from EDMDSimulator import EDMDSimulator
import numpy as np

# 시스템 정의
def fwd_nonlinear_damping(t, x, u=0, dt=0.1, k=1.0, gamma=1.0):
    """
    비선형 감쇠 오실레이터의 Euler 이산화를 수행하는 함수
    
    dx1/dt = x2
    dx2/dt = -k * x1 - gamma * x2^3

    Parameters:
        t  : 현재 시간 (사용 안 함, 단일 입력)
        x  : 상태 (2 x N numpy array), [x1; x2] 형태
        u  : 외부 입력 (미사용)
        dt : 이산 시간 간격
        k  : 스프링 계수
        gamma : 비선형 감쇠 계수

    Returns:
        다음 상태를 나타내는 2 x N numpy array
    """
    x1, x2 = x[0, :], x[1, :]
    dx1 = x1 + x2 * dt
    dx2 = x2 + (-k * x1 - gamma * x2**3) * dt
    return np.array([dx1, dx2])

def setup_basis_functions():
    """
    Basis functions of Koopman operator eigenfunctions.
    """
    centers = np.random.uniform(-2,2,(2,100))

    RFF = observables.RandomFourierFeatures(
        include_state=True,
        gamma=1.0,
        D=100,
        random_state=42    # 시드
        )
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
        degree=7,
        include_bias=True
    )

    return RFF, RBF_thinplate, RBF_gauss, RBF_invquad, RBF_invmultquad, RBF_polyharmonic, POLY

def main():
    RFF, RBF_thinplate, RBF_gauss, RBF_invquad, RBF_invmultquad, RBF_polyharmonic, POLY = setup_basis_functions()

    sim = EDMDSimulator(
    system_name='duffing',
    system_function=fwd_nonlinear_damping,
    n_states=2,
    dT=0.1,
    observable = RFF
    )

    x0 = np.array([0.3, -0.5])

    sim.generate_training_data(xrange=(-0.5, 0.5), yrange=(-0.5, 0.5), grid_points=100)
    sim.fit_model()
    sim.evaluate_model(x0, step = 200)
    sim.print_single_trajectory_errors()
    sim.plot_phase_portrait(x0, xlim=(-1, 1), ylim=(-1, 1), grid_points=30)

if __name__ == "__main__":
    main()