import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans_example():
    print("--- 1. Chargement et Préparation des Données (Iris) ---")
    # Chargement du dataset Iris (150 échantillons, 4 features)
    iris = datasets.load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    # Normalisation : StandardScaler met chaque feature à N(0,1)
    scaler = StandardScaler()
    X_iris_s = scaler.fit_transform(X_iris)
    
    # Réduction de dimension pour la visualisation (PCA 2D)
    pca_iris = PCA(n_components=2)
    X_iris_2d = pca_iris.fit_transform(X_iris_s)
    
    print("--- 2. Méthode Elbow : Trouver le k optimal ---")
    inertias = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_iris_s)
        inertias.append(km.inertia_)
        
    # Optionnel: Tracer la courbe Elbow
    # plt.plot(k_range, inertias, marker='o')
    # plt.title('Méthode Elbow')
    # plt.xlabel('Nombre de clusters (k)')
    # plt.ylabel('Inertie (WCSS)')
    # plt.show()
    print("Le 'coude' de l'inertie indique généralement k=3 pour le dataset Iris.")

    print("\n--- 3. Entraînement de K-Means (k=3) ---")
    # Entraînement de K-Means avec k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(X_iris_s)
    
    print("\n--- 4. Évaluation des Résultats ---")
    # Calcul du score de Silhouette
    sil = silhouette_score(X_iris_s, labels_km)
    print(f"Score de Silhouette : {sil:.3f} (attendu: ~0.46)")
    print(f"Inertie (WCSS)      : {kmeans.inertia_:.2f} (attendu: ~139)")
    
    # Projection des centroïdes en 2D pour la visualisation
    centers_2d = pca_iris.transform(kmeans.cluster_centers_)
    
    # Visualisation des clusters en 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=labels_km, cmap='viridis', marker='o', alpha=0.6, label='Points de données')
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=200, label='Centroïdes K-Means')
    plt.title("Clustering K-Means sur le dataset Iris (PCA 2D)")
    plt.xlabel("Composante Principale 1")
    plt.ylabel("Composante Principale 2")
    plt.legend()
    # plt.show() # Décommenter pour afficher le graphique
    print("\nL'exécution est terminée. Décommentez plt.show() pour voir les graphiques.")

if __name__ == "__main__":
    run_kmeans_example()
