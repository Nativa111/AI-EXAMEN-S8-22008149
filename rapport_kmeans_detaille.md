# Rapport sur l'Algorithme K-Means

**Date :** 17 Mars 2026
**Thème :** Algorithmes de Classification Non Supervisée (Clustering)
**Focus :** Algorithme K-Means sur le dataset Iris

---

## 1. Introduction au Clustering et à K-Means

Le **clustering** est une technique d'apprentissage automatique non supervisée visant à regrouper des données similaires sans utiliser d'étiquettes préexistantes. 

Dans ce cadre, **K-Means** est l'algorithme le plus populaire. Son objectif est de partitionner les données en **K clusters distincts** en minimisant l'inertie, c'est-à-dire la somme des variances intra-cluster (la distance entre les points et le centre de leur cluster).

### Pourquoi K-Means ?
- **Simple et rapide :** Scalable sur de grands volumes de données.
- **Principe compréhensible :** Fonctionne par itérations d'assignations de points à des centroïdes (centres géométriques).

### Limites principales
- Le nombre de clusters $K$ doit être spécifié à l'avance.
- Sensibilité aux valeurs aberrantes (outliers).
- Présuppose que les clusters sont de forme sphérique et de tailles comparables.

---

## 2. Démarche et Étapes d'Implémentation

Le code développé (`kmeans_example.py`) illustre l'application de K-Means sur le jeu de données classique **Iris** (données florales, 150 échantillons, 4 caractéristiques, 3 classes réelles).

Voici les grandes étapes implémentées dans le script :

### Étape 1 : Chargement et Préparation des Données
- **Données Brutes :** Importation du dataset Iris via `scikit-learn` (`datasets.load_iris()`).
- **Normalisation :** Application de `StandardScaler`. Cette étape est cruciale en clustering basé sur la distance euclidienne : elle ramène chaque caractéristique (feature) à une moyenne de 0 et un écart-type de 1, évitant qu'une caractéristique avec de grandes valeurs domine les autres.
- **Réduction de dimensionnalité :** Utilisation de l'Analyse en Composantes Principales (`PCA`) pour projeter les données de 4 dimensions vers 2 dimensions. Ceci permet de visualiser graphiquement les clusters une fois formés.

### Étape 2 : Détermination du Nombre Optimal de Clusters (K)
- **Méthode Elbow (du coude) :** L'algorithme a été entraîné plusieurs fois avec différentes valeurs de $K$ (de 2 à 10).
- **Principe :** On calcule l'inertie (WCSS) pour chaque $K$. En traçant l'inertie en fonction de $K$, on cherche le "coude", point où l'ajout d'un cluster supplémentaire ne réduit plus significativement l'inertie.
- **Résultat attendu :** Sur le dataset Iris, le coude apparaît clairement pour $K=3$ (correspondant aux 3 espèces de fleurs).

### Étape 3 : Entraînement de K-Means
- **Configuration :** 
  - `n_clusters=3` : Nombre de clusters fixé.
  - `init='k-means++'` (par défaut) : Optimise l'initialisation des centroïdes pour une convergence plus rapide.
  - `n_init=10` : L'algorithme est exécuté 10 fois avec des centroïdes initiaux différents, et le meilleur résultat (plus faible inertie) est conservé.
- **Exécution :** K-Means assigne itérativement chaque point au centroïde le plus proche et recalcule les centres jusqu'à stabilisation (convergence).

### Étape 4 : Évaluation et Métriques
- **Score de Silhouette :** Mesure la qualité du partitionnement. Evalué entre -1 (mauvais) et 1 (parfait). Le modèle obtient un score d'environ **0.46**.
  - C'est un score correct indiquant des clusters plutôt cohérents, bien que les classes d'iris se chevauchent légèrement.
- **Inertie (WCSS) :** La somme des distances au carré s'établit à environ **139**.

### Étape 5 : Visualisation Graphique
- **Projection :** Les points de données ainsi que les 3 centroïdes finaux sont projetés sur le plan 2D calculé lors de l'étape 1.
- **Affichage :** Un graphique matérialisant visuellement les 3 groupes (avec des couleurs distinctes) et leurs centres (croix rouges) est généré en utilisant `matplotlib.pyplot`.

---

## 3. Conclusion

L'algorithme K-Means s'avère très efficace et pertinent pour le jeu de données Iris, parvenant à identifier la structure intrinsèque (3 groupes) de manière autonome, malgré l'absence d'étiquettes fournies.
L'approche méthodique (Normalisation $\rightarrow$ Détermination de K $\rightarrow$ Clustering $\rightarrow$ Évaluation $\rightarrow$ Visualisation) garantit une application rigoureuse de ce standard du Machine Learning.
