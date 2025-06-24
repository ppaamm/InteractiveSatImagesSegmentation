import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from . import optimization


def alignment_score(confusion_matrix):
    try:
        row_maxes = np.max(confusion_matrix, axis=1)
        return np.mean(row_maxes)
    except:
        return 0


class HyperparameterSelection(ABC):
    
    def __init__(self, X):
        self.X = X
        self.scaler = StandardScaler().fit(X)
        self.X_scaled = self.scaler.transform(X)
        self.current_step = 0
            
    @abstractmethod
    def _optimize_parameter(self, alignment):
        pass
    
        
    def next_step(self, confusion_matrix):
        """Perform one step of segmentation with increasing complexity."""
        self.current_step += 1
        alignment = alignment_score(confusion_matrix)
        return self._optimize_parameter(alignment)
        
        
        
        

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

