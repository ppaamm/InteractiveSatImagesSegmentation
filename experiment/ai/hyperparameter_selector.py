import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from itertools import product
from abc import ABC, abstractmethod
from . import optimization
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def alignment_score(confusion_matrix):
    try:
        row_maxes = np.max(confusion_matrix, axis=1)
        return np.mean(row_maxes)
    except:
        return 0



###############################################################################
###                             BASE SELECTION                              ###
###############################################################################

def label_difference(y1, y2):
    cm = confusion_matrix(y1, y2)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize matching
    matched = cm[row_ind, col_ind].sum()
    total = len(y1)
    diff_ratio = 1 - matched / total
    return diff_ratio


class HyperparameterSelection(ABC):
    
    def __init__(self, X, max_n_attempt=5):
        self.X = X
        self.scaler = StandardScaler().fit(X)
        self.X_scaled = self.scaler.transform(X)
        self.current_step = 0
        self.max_n_attempt = 5
        self.history = []
            
    @abstractmethod
    def _optimize_parameter(self, alignment):
        pass
    
    
    def _already_found_clustering(self, labels):
        print("Entering already found clustering function")
        for previous_step in self.history:
            old_labels = previous_step['labels']
            difference = label_difference(labels, old_labels)
            print("Difference:", difference)
            if difference == 0:
                print("Already found" )
                return True, previous_step['confusion_matrix']
        
        return False, None
    
    def next_step(self, confusion_matrix, n_attempt=0):
        """Perform one step of segmentation with increasing complexity."""
        
        print("===============  Entering next step ===============")
        print("History:", len(self.history))
        
        if n_attempt == 0 and self.history:
            print("Adds the previous confusion matrix")
            self.history[-1]['confusion_matrix'] = confusion_matrix
        
        
        self.current_step += 1
        alignment = alignment_score(confusion_matrix)
        print("Alignment score:", alignment)
        
        predicted_labels = self._optimize_parameter(alignment)
        print("Predicted new labels")
        
        already_found, found_matrix = self._already_found_clustering(predicted_labels)
        
        if already_found and n_attempt < self.max_n_attempt:
            self.current_step -= 1
            return self.next_step(found_matrix, n_attempt+1)
        
        self.history.append({'labels': predicted_labels})
        return predicted_labels
        
        
        
        
###############################################################################
###                                K MEANS                                  ###
###############################################################################
        

class BasicKMeans(HyperparameterSelection):
    def _optimize_parameter(self, alignment):
        n_clusters = 1 + self.current_step  # Just an example progression

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.X_scaled)
        
        print("worked")
        return labels
    


class KMeansOptimizer(HyperparameterSelection):
    def __init__(self, X, K_max = 10, noise_variance=0.1):
        super().__init__(X)
        search_space = np.arange(2, K_max + 1)
        gp = optimization.GaussianProcess(optimization.squared_exponential_kernel, noise_variance)
        self.bo = optimization.BasicBO(gp, optimization.ucb, search_space)
        self.current_K = 0
        
    def _optimize_parameter(self, alignment):
        if self.current_step == 1:
            self.current_K = 2
        else:
            self.bo.update_observations(self.current_K, alignment)
            self.current_K = self.bo.select_next()[0][0]
        
        print(self.current_K)
        kmeans = KMeans(n_clusters=self.current_K, random_state=42)
        return kmeans.fit_predict(self.X_scaled)



# def kmeans_mahalanobis(X, n_clusters, M_diag, random_state=42):
#     M = np.diag(M_diag)
#     X_transformed = X @ np.linalg.cholesky(M).T  # linear transformation
#     km = KMeans(n_clusters=n_clusters, random_state=random_state)
#     labels = km.fit_predict(X_transformed)
#     return labels


# class KMeansMahalanobisOptimizer(HyperparameterSelection):
#     def __init__(self, X, K_max=10, noise_variance=0.1):
#         super().__init__(X)
        
#         print("Dataset:", X.shape)

#         self.d = self.X_scaled.shape[1]

#         K_values = np.arange(2, K_max + 1)
#         diag_values = [0.1, 1.0]  # example scale options for Mahalanobis diag
#         search_space = []

#         for K in K_values:
#             for diag in product(diag_values, repeat=self.d):
#                 search_space.append((K,) + diag)

#         search_space = np.array(search_space)
#         print("Search space:", search_space.shape)

#         gp = optimization.GaussianProcess(optimization.squared_exponential_kernel, noise_variance)
#         self.bo = optimization.BasicBO(gp, optimization.ucb, search_space)
#         self.current_params = None

    

#     def _optimize_parameter(self, alignment):
#         if self.current_step == 1:
#             self.current_params = self.bo.search_space[0]
#         else:
#             self.bo.update_observations(self.current_params, alignment)
#             self.current_params = self.bo.select_next()[0]
    
#         K = int(self.current_params[0])
#         M_diag = np.array(self.current_params[1:])
#         print(f"K={K}, M_diag={M_diag}")
    
#         return kmeans_mahalanobis(self.X_scaled, K, M_diag)




def transform_with_mahalanobis(X, group_diag):
    """
    Apply a linear transformation based on a grouped diagonal Mahalanobis matrix.
    """
    feature_groups = [3, 3, 2, X.shape[1] - 8]
    full_diag = np.concatenate([
        np.full(size, scale) for scale, size in zip(group_diag, feature_groups)
    ])
    M = np.diag(full_diag)
    L = np.linalg.cholesky(M)
    return X @ L.T


def kmeans_mahalanobis(X, n_clusters, group_diag, random_state=42):
    X_transformed = transform_with_mahalanobis(X, group_diag)
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X_transformed)
    return labels


class KMeansMahalanobisOptimizer(HyperparameterSelection):
    def __init__(self, X, K_max=10, noise_variance=0.1):
        super().__init__(X)

        print("Dataset:", X.shape)
        self.d = self.X_scaled.shape[1]
        assert self.d >= 9, "This grouping assumes at least 9 features"

        K_values = np.arange(2, K_max + 1)
        group_diag_values = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]  # small discrete grid
        search_space = []

        for K in K_values:
            for diag in product(group_diag_values, repeat=4):  # only 4 parameters now
                search_space.append((K,) + diag)

        search_space = np.array(search_space)
        print("Search space:", search_space.shape)

        gp = optimization.GaussianProcess(optimization.squared_exponential_kernel, noise_variance)
        self.bo = optimization.BasicBO(gp, optimization.ucb, search_space)
        self.current_params = None

    def _optimize_parameter(self, alignment):
        if self.current_step == 1:
            self.current_params = self.bo.search_space[0]
        else:
            self.bo.update_observations(self.current_params, alignment)
            self.current_params = self.bo.select_next()[0]

        K = int(self.current_params[0])
        group_diag = np.array(self.current_params[1:])
        print(f"K={K}, group_diag={group_diag}")

        return kmeans_mahalanobis(self.X_scaled, K, group_diag)








###############################################################################
###                                 DBSCAN                                  ###
###############################################################################


class DBSCANOptimizer(HyperparameterSelection):
    def __init__(self, X, eps_range=(0.1, 5.0), min_samples_range=(3, 10), n_eps=20, 
                 noise_variance=0.1, max_clusters=10):
        super().__init__(X)
        
        self.max_clusters = max_clusters

        eps_values = np.linspace(eps_range[0], eps_range[1], n_eps)
        min_samples_values = np.arange(min_samples_range[0], min_samples_range[1] + 1)
        search_space = np.array(list(product(eps_values, min_samples_values)))
        
        gp = optimization.GaussianProcess(optimization.squared_exponential_kernel, noise_variance)
        self.bo = optimization.BasicBO(gp, optimization.ucb, search_space)
        self.current_params = None

    def _optimize_parameter(self, alignment):
        if self.current_step == 1:
            self.current_params = self.bo.search_space[len(self.bo.search_space) // 2]
        else:
            self.bo.update_observations(self.current_params, alignment)

        # Try selecting a valid next configuration
        tried = set()
        while True:
            if self.current_step != 1:
                self.current_params = self.bo.select_next()[0]
            param_tuple = tuple(self.current_params)
            if param_tuple in tried:
                raise RuntimeError("No suitable DBSCAN configuration found with acceptable number of clusters.")
            tried.add(param_tuple)

            eps, min_samples = self.current_params
            dbscan = DBSCAN(eps=eps, min_samples=int(min_samples))
            labels = dbscan.fit_predict(self.X_scaled)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters <= self.max_clusters:
                print(f"DBSCAN step {self.current_step}: eps={eps:.3f}, min_samples={int(min_samples)}, clusters={n_clusters}")
                return labels
            else:
                print(f"Rejected params (too many clusters: {n_clusters}): eps={eps:.3f}, min_samples={int(min_samples)}")


###############################################################################
###                          Spectral clustering                            ###
###############################################################################


class SpectralClusteringOptimizer(HyperparameterSelection):
    def __init__(self, X, K_max=10, gamma_values=None, noise_variance=0.1):
        super().__init__(X)

        if gamma_values is None:
            # Log-spaced values from 0.1 to 10
            #gamma_values = np.logspace(-5, 1, 5)  # e.g., [0.1, 0.316, 1, 3.16, 10]
            gamma_values = [5, 10, 20, 50]

        K_values = np.arange(2, K_max + 1)
        search_space = np.array(list(product(K_values, gamma_values)))  # shape (n_combinations, 2)

        gp = optimization.GaussianProcess(optimization.squared_exponential_kernel, noise_variance)
        self.bo = optimization.BasicBO(gp, optimization.ucb, search_space)
        self.current_params = None

    def _optimize_parameter(self, alignment):
        if self.current_step == 1:
            self.current_params = self.bo.search_space[0]
        else:
            self.bo.update_observations(self.current_params, alignment)
            self.current_params = self.bo.select_next()[0]

        K = int(self.current_params[0])
        gamma = float(self.current_params[1])

        print(f"Spectral Clustering step {self.current_step}: K={K}, gamma={gamma:.3f}")

        # clustering = SpectralClustering(
        #     n_clusters=K,
        #     affinity='rbf',
        #     gamma=gamma,
        #     assign_labels='kmeans',
        #     random_state=42
        # )
        
        clustering = SpectralClustering(
            n_clusters=K,
            affinity='nearest_neighbors',
            n_neighbors=gamma,  # try 5–20 depending on dataset size
            assign_labels='kmeans',
            random_state=42
        )
        return clustering.fit_predict(self.X_scaled)


class SpectralMahalanobisOptimizer(HyperparameterSelection):
    def __init__(self, X, K_max=10, noise_variance=0.1):
        super().__init__(X)

        self.d = self.X_scaled.shape[1]
        assert self.d >= 9, "Mahalanobis grouping expects at least 9 features."

        # Search space grid values
        K_values = np.arange(2, K_max + 1)
        gamma_values = np.logspace(-6, -1, 5)         # [0.1, 0.316, 1, 3.16, 10]
        group_diag_values = [0.01, 0.5, 1.0]               # Values for each group in the Mahalanobis diagonal

        # Build search space: (K, gamma, d1, d2, d3, d4)
        search_space = []
        for K in K_values:
            for gamma in gamma_values:
                for diag in product(group_diag_values, repeat=4):
                    if all(v > 0 for v in diag):
                        search_space.append((K, gamma) + diag)

        search_space = np.array(search_space)

        gp = optimization.GaussianProcess(optimization.squared_exponential_kernel, noise_variance)
        self.bo = optimization.BasicBO(gp, optimization.ucb, search_space)
        self.current_params = None

    def _optimize_parameter(self, alignment):
        if self.current_step == 1:
            self.current_params = self.bo.search_space[0]
        else:
            self.bo.update_observations(self.current_params, alignment)
            self.current_params = self.bo.select_next()[0]

        K = int(self.current_params[0])
        gamma = float(self.current_params[1])
        group_diag = np.array(self.current_params[2:])

        print(f"Step {self.current_step}: K={K}, gamma={gamma:.3f}, group_diag={group_diag}")

        X_transformed = transform_with_mahalanobis(self.X_scaled, group_diag)

        clustering = SpectralClustering(
            n_clusters=K,
            affinity='rbf',
            gamma=gamma,
            assign_labels='kmeans',
            random_state=42
        )

        return clustering.fit_predict(X_transformed)