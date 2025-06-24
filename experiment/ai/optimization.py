import numpy as np


###############################################################################
###                                 KERNELS                                 ###
###############################################################################

def squared_exponential_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """
    Squared Exponential (RBF) kernel function.

    Parameters:
        x1, x2 : numpy arrays
            Input points for the kernel function.
        length_scale : float
            Controls the smoothness of the function.
        variance : float
            Vertical variation (amplitude).

    Returns:
        Kernel matrix between x1 and x2.
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return variance * np.exp(-0.5 * sqdist / length_scale**2)



###############################################################################
###                          ACQUISITION FUNCTIONS                          ###
###############################################################################

def ucb(mu, sigma, beta=2.0):
    return mu + beta * sigma


###############################################################################
###                            GAUSSIAN PROCESS                             ###
###############################################################################


class GaussianProcess:
    def __init__(self, kernel, noise_variance):
        self.kernel = kernel
        self.noise_variance = noise_variance
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.ravel()
        
        n = X.shape[0]
        K = self.kernel(X, X) + self.noise_variance * np.eye(n)
        K += 1e-8 * np.eye(n)
        
        self.K_inv = np.linalg.inv(K)
        
        
    def predict_full(self, X_test):
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test)
        
        mu_s = K_s.T @ self.K_inv @ self.y_train  
        cov_s = K_ss - K_s.T @ self.K_inv @ K_s 
        return mu_s, cov_s
    
    def predict(self, X_test):
        K_s = self.kernel(self.X_train, X_test)  # shape (n_train, n_test)
        mu_s = K_s.T @ self.K_inv @ self.y_train
        mu_s = mu_s.ravel()
    
        # Efficiently compute only the diagonal of the covariance (variances)
        v = self.K_inv @ K_s  # shape (n_train, n_test)
        K_ss_diag = np.array([self.kernel(x, x)[0,0] for x in X_test])
        var_s = K_ss_diag - np.sum(K_s * v, axis=0)  # element-wise dot product per test point
    
        return mu_s, var_s




###############################################################################
###                          BAYESIAN OPTIMIZATION                          ###
###############################################################################


class BasicBO:
    def __init__(self, GP, acquisition_function, search_space):
        self.GP = GP
        self.acquisition_function = acquisition_function
        
        search_space = np.atleast_2d(search_space)
        if search_space.shape[0] < search_space.shape[1]:
            search_space = search_space.T
        self.search_space = search_space

        
        self.X_obs = None
        self.y_obs = None
        
    
    def update_observations(self, x_new, y_new):
        if self.X_obs is None:
            self.X_obs = np.array([x_new])
            self.y_obs = np.array([y_new])
        else:
            self.X_obs = np.vstack([self.X_obs, x_new])
            self.y_obs = np.append(self.y_obs, y_new)
            
        self.GP.fit(self.X_obs, self.y_obs)
    
    def select_next(self):
        # if self.X_obs is not None:
        #     mask = ~np.any(np.all(self.search_space[:, None] == self.X_obs[None, :], axis=2), axis=1)
        #     candidate_points = self.search_space[mask]
        # else:
        #     candidate_points = self.search_space
    
        candidate_points = self.search_space
    
        if candidate_points.shape[0] == 0:
            raise ValueError("All points in the search space have already been evaluated.")
    
        # Predict using GP
        mu, cov = self.GP.predict(candidate_points)
        
        if cov.ndim == 2:
            sigma = np.sqrt(np.diag(cov))
        else:
            sigma = np.sqrt(cov)
            
        print("Search space:", candidate_points.shape)
        print("mu:", mu.shape)
        print("sigma:", cov.shape)

        acq_values = self.acquisition_function(mu, sigma)
        
        print("Acquisition values:", acq_values.shape)
        
        best_idx = np.argmax(acq_values)
        return candidate_points[best_idx], acq_values[best_idx]