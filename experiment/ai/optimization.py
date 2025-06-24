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
        
        
    def predict(self, X_test):
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test)
        
        mu_s = K_s.T @ self.K_inv @ self.y_train  
        cov_s = K_ss - K_s.T @ self.K_inv @ K_s 
        return mu_s, cov_s



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
        mu, cov = self.GP.predict(self.search_space)
        print(mu.shape, cov.shape)
        
        if cov.ndim == 2:
            sigma = np.sqrt(np.diag(cov))
        else:
            sigma = np.sqrt(cov)

        acq_values = self.acquisition_function(mu, sigma)
        best_idx = np.argmax(acq_values)
        return self.search_space[best_idx], acq_values[best_idx]