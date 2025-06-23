import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod


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
    

        

