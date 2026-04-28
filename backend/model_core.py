import numpy as np;
from sklearn.decomposition import PCA;

class FaceRecognitionModel:
  def __init__(self, n_components=150):
    self.n_components = n_components
    self.pca = PCA(n_components=n_components, whiten=True, svd_solver='randomized')
    self.mean_face = None
    self.eigenfaces = None
    self.projections = []
    self.names = []

  def train(self, X, names):
        """
        X: Matrix of flattened grayscale images (num_images, num_pixels)
        names: List of names corresponding to images
        """
       
        self.pca.fit(X)
        self.mean_face = self.pca.mean_
        self.eigenfaces = self.pca.components_
        
   
        self.projections = self.pca.transform(X)
        self.names = names
        print(f"Training complete. Found {len(self.eigenfaces)} Eigenfaces.")

  def recognize(self, test_face_vector):
        """
        Projects a new face and finds the closest match using Euclidean Distance.
        """
   
        query_projection = self.pca.transform([test_face_vector])
        
     
        distances = np.linalg.norm(self.projections - query_projection, axis=1)
        best_match_idx = np.argmin(distances)
        
        return self.names[best_match_idx], distances[best_match_idx]