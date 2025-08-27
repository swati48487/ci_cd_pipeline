#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

#load data
df = pd.read_csv(r"C:\Users\Swati Singh\Downloads\Mall_Customers.csv")

#preprocessing
df = df.drop(columns=["CustomerID"], errors='ignore')
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])  # Converts 'Male'/'Female' to 0/1
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Genre']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

#fit model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Predict cluster labels for test set
labels = kmeans.predict(X_scaled)

#save pkl file
import joblib
joblib.dump(kmeans, "kmean_mall.pkl")