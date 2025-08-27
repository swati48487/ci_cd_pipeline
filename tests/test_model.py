
import unittest
import joblib
from sklearn.cluster import KMeans

class TestKMeansModel(unittest.TestCase):
    def test_kmeans_model(self):
        kmeans = joblib.load('kmean_mall.pkl')
        self.assertIsInstance(kmeans, KMeans)
        self.assertGreaterEqual(len(kmeans.cluster_centers_), 3)

if __name__ == '__main__':
    unittest.main()