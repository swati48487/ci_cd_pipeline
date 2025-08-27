#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

#load the saved model
kmeans=joblib.load('kmean_mall.pkl')

#make predictions
labels = kmeans.predict(X_scaled)

#evaluate
score = silhouette_score(X_scaled, labels)
print("Silhouette Score on Test Set:", score)

