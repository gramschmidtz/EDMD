import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import pykoopman as pk

class EDMDSimulator:
    def __init__(self, system_name, system_function, n_states=2, dT=0.1, n_traj=50, n_int=10,
                 futuresize=2.0, observable=None, random_seed=42):
        self.system_name = system_name
        self.system_function = system_function
        self.n_states = n_states
        self.dT = dT
        self.n_traj = n_traj
        self.n_int = n_int
        self.futuresize = futuresize
        self.observable = observable
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def generate_training_data(self, xrange=(-2, 2), yrange=(-2, 2), grid_points=100):
        """
        Generate training data using meshgrid over given (xrange, yrange),
        then apply one-step Duffing integration to form (X, Y) data for EDMD.
        """
        # 1. 격자 생성
        x1 = np.linspace(xrange[0], xrange[1], grid_points)
        x2 = np.linspace(yrange[0], yrange[1], grid_points)
        X1, X2 = np.meshgrid(x1, x2)

        # 2. 상태 벡터 (2, N) 형태로 정렬
        xE = np.vstack([X1.ravel(), X2.ravel()])  # shape: (2, grid_points^2)
        self.n_traj = xE.shape[1]  # 총 trajectory 수 업데이트

        # 3. 각 점에 대해 한 스텝 forward integration
        x = xE.copy()
        y = self.system_function(0, x, dt=self.dT)  # shape: (2, N)

        # 4. 학습용 데이터 저장
        self.X = x          # 입력 상태: shape (2, N)
        self.Y = y          # 다음 상태: shape (2, N)
        self.xE = xE        # initial condition 저장 (예측용)


    def fit_model(self):
        EDMD = pk.regression.EDMD()
        self.model = pk.Koopman(observables=self.observable, regressor=EDMD)
        self.model.fit(self.X.T, y=self.Y.T)
        pd.DataFrame(self.model.A).to_excel(f'koopman_matrix_{self.system_name}.xlsx', index=False)

    def evaluate_model(self, x0, step=100):
        """
        Evaluate Koopman prediction and ground truth from initial state x0 over given step length.
        """
        t = np.arange(0, step * self.dT, self.dT)

        # Ground truth trajectory
        Xtrue = np.zeros((len(t), self.n_states))
        Xtrue[0] = x0
        for i in range(1, len(t)):
            y = self.system_function(0, Xtrue[i - 1, :][:, np.newaxis], dt=self.dT)
            Xtrue[i, :] = y.ravel()

        # Koopman prediction
        Xkoop = self.model.simulate(x0, n_steps=step - 1)
        Xkoop = np.vstack([x0[np.newaxis, :], Xkoop])

        # Store results
        self.t = t
        self.Xtrue = Xtrue
        self.Xkoop = Xkoop
        self.mean_error = np.mean(np.linalg.norm(Xtrue - Xkoop, axis=1))
        self.max_error = np.max(np.linalg.norm(Xtrue - Xkoop, axis=1))

    def predict_on_training(self):
        x = self.xE.T
        yT = self.model.predict(x).T
        self.Xk = x.T
        self.Yk = yT

    def print_single_trajectory_errors(self):
        """
        Print the Koopman model prediction error (mean and max L2 norms).
        """
        print(f"✅ Koopman Prediction Mean L2 Error: {self.mean_error:.6f}")
        print(f"🚨 Max L2 Error over trajectory:     {self.max_error:.6f}")

    def plot_results(self, x0):
        # Time-series plot
        fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True, figsize=(10, 4))
        axs[0].plot(self.t, self.Xtrue[:, 0], '-', color='b', label='True')
        axs[0].plot(self.t, self.Xkoop[:, 0], '--r', label='EDMD')
        axs[1].plot(self.t, self.Xtrue[:, 1], '-', color='b', label='True')
        axs[1].plot(self.t, self.Xkoop[:, 1], '--r', label='EDMD')
        axs[1].set(xlabel=r'$t$')
        axs[0].set(ylabel=r'$x_1$')
        axs[1].set(ylabel=r'$x_2$')
        for ax in axs:
            ax.legend()

        # Phase plot
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 6))
        for traj_idx in range(self.n_traj):
            ax.plot([self.X[0, traj_idx::self.n_traj], self.Y[0, traj_idx::self.n_traj]],
                    [self.X[1, traj_idx::self.n_traj], self.Y[1, traj_idx::self.n_traj]], '-ok', alpha=0.9, markersize=4)
            ax.plot([self.Xk[0, traj_idx::self.n_traj], self.Yk[0, traj_idx::self.n_traj]],
                    [self.Xk[1, traj_idx::self.n_traj], self.Yk[1, traj_idx::self.n_traj]], '--r', alpha=0.9)

        ax.plot(self.Xtrue[:, 0], self.Xtrue[:, 1], '-^b')
        ax.plot(self.Xkoop[:, 0], self.Xkoop[:, 1], '->c')
        ax.scatter(x0[0], x0[1], s=50, c='b', label='unseen test')
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_title('red: train - pred, black: train - original, blue: unseen test')

        red_patch = mpatches.Patch(color='red', label='training prediction')
        black_patch = mpatches.Patch(color='black', label='training ground truth')
        blue_patch = mpatches.Patch(color='blue', label='unseen test trajectory')
        cyan_patch = mpatches.Patch(color='cyan', label='Koopman prediction on test')
        ax.legend(handles=[red_patch, black_patch, blue_patch, cyan_patch], loc='best')
        plt.show()

    def plot_phase_portrait(self, x0, xlim=[-2.5, 2.5], ylim=[-2.5, 2.5], grid_points=30):
        """
        Draw Duffing oscillator phase portrait and overlay Koopman prediction vs true trajectory.
        Adds red rectangle showing training domain if available.
        """
        # 1. Mesh grid for vector field
        x1 = np.linspace(xlim[0], xlim[1], grid_points)
        x2 = np.linspace(ylim[0], ylim[1], grid_points)
        X1, X2 = np.meshgrid(x1, x2)
        X_flat = np.vstack([X1.ravel(), X2.ravel()])

        # 2. Vector field approximation
        X_next = self.system_function(0, X_flat, dt=self.dT)
        DX = (X_next - X_flat) / self.dT
        U = DX[0, :].reshape(grid_points, grid_points)
        V = DX[1, :].reshape(grid_points, grid_points)
        N = np.sqrt(U**2 + V**2)
        U_unit = U / N
        V_unit = V / N

        # 3. Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        q = ax.quiver(X1, X2, U_unit, V_unit, N, cmap='viridis', scale=40, pivot='mid', alpha=0.8)
        cbar = fig.colorbar(q, ax=ax)
        cbar.set_label("Vector magnitude")

        # 4. Draw training data box if available
        if hasattr(self, "X"):
            x_min, x_max = np.min(self.X[0, :]), np.max(self.X[0, :])
            y_min, y_max = np.min(self.X[1, :]), np.max(self.X[1, :])
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                edgecolor='red', facecolor='none',
                                linewidth=2, linestyle='--', label='training domain')
            ax.add_patch(rect)

        # 5. Trajectories
        ax.plot(self.Xtrue[:, 0], self.Xtrue[:, 1], '-^b', label='unseen test trajectory')
        ax.plot(self.Xkoop[:, 0], self.Xkoop[:, 1], '->c', label='Koopman prediction on test')
        ax.scatter(x0[0], x0[1], s=50, c='b')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_title("Duffing Phase Portrait + Koopman Prediction")
        ax.grid(True)
        ax.legend()
        plt.show()