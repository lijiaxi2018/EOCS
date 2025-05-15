import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class DVFSController:
    def __init__(self, cpu_frequency_list, gpu_frequency_list):
        self.cpu_frequency_list = cpu_frequency_list
        self.gpu_frequency_list = gpu_frequency_list

        # Training data
        self.X = []  # (cpu_freq, gpu_freq)
        self.y_fps_od = []
        self.y_fps_p = []
        self.y_tps = []
        self.y_power = []

        # Gaussian Process models
        kernel = RBF(length_scale=1.0)
        self.gp_fps_od = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.gp_fps_p = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.gp_tps = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.gp_power = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

        self.is_fitted = False

    def update(self, cpu_freq, gpu_freq, fps_od, fps_p, tps, power):
        # Add new training data
        self.X.append([cpu_freq, gpu_freq])
        self.y_fps_od.append(fps_od)
        self.y_fps_p.append(fps_p)
        self.y_tps.append(tps)
        self.y_power.append(power)

        # Fit all models
        X_train = np.array(self.X)
        self.gp_fps_od.fit(X_train, self.y_fps_od)
        self.gp_fps_p.fit(X_train, self.y_fps_p)
        self.gp_tps.fit(X_train, self.y_tps)
        self.gp_power.fit(X_train, self.y_power)

        self.is_fitted = True

    def tell(self, required_fps_od, required_fps_p, required_tps):
        if not self.is_fitted:
            return max(self.cpu_frequency_list), max(self.gpu_frequency_list)

        # Evaluate all combinations
        candidates = []
        for cpu_freq in self.cpu_frequency_list:
            for gpu_freq in self.gpu_frequency_list:
                x = np.array([[cpu_freq, gpu_freq]])

                pred_fps_od = self.gp_fps_od.predict(x)[0]
                pred_fps_p = self.gp_fps_p.predict(x)[0]
                pred_tps = self.gp_tps.predict(x)[0]
                pred_power = self.gp_power.predict(x)[0]

                if (pred_fps_od >= required_fps_od and
                    pred_fps_p >= required_fps_p and
                    pred_tps >= required_tps):
                    candidates.append((pred_power, cpu_freq, gpu_freq))

        if not candidates:
            return max(self.cpu_frequency_list), max(self.gpu_frequency_list)

        # Choose the one with minimal power
        candidates.sort()
        _, best_cpu, best_gpu = candidates[0]
        return best_cpu, best_gpu