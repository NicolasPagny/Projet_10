############################################################################################################
# Ce fichier contient des fonctions qui permet d'effectuer des réductions de dimensions ou 
# des fonctions qui précalculent certains opérations afin d'économiser les ressources et d'accélérer 
# le temps de calcul des systèmes de recommandations
############################################################################################################

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler

def most_popular_articles(clicks: pd.DataFrame, n: int = 5):
    """Obtenir les articles les plus populaires
    
    Parameters:
    clicks: la dataframe contenant les clics
    n: le nombre d'articles à retourner
    """

    
    # Compter le nombre de clics par article
    nb_clicks = clicks["click_article_id"].value_counts()
    nb_clicks = nb_clicks.sort_values(ascending=False)

    # Sélectionner les n articles les plus populaires
    most_popular_articles = nb_clicks.head(n)

    return most_popular_articles
     

def pca_reduct_articles_embeddings(df_embedding_articles: pd.DataFrame, df_clicks_hour: pd.DataFrame, n=0.95):
        """
        Filtre le dataset articles_embeddings afin de ne garder que les articles susceptibles d'intéresser les clients
        et ensuite appliquer une réduction de dimension par PCA

        Parameters:
        df_embedding_articles: l'embeddings des articles original
        df_clicks_hour: les interactions utilisateurs-articles

        Return:
        Une dataframe contenant l'embeddings des articles réduits
        """

        #On filtre les articles consultés
        unique_article_id = df_clicks_hour["click_article_id"].unique()
        df_reducted_embedding_articles = df_embedding_articles.loc[df_embedding_articles.index.isin(unique_article_id)]

        #On applique une réduction de dimension par PCA
        pca = PCA(n_components=n)
        df_pca_reducted_embedding_articles = pd.DataFrame(pca.fit_transform(df_reducted_embedding_articles))
        df_pca_reducted_embedding_articles.index = df_reducted_embedding_articles.index

        return df_pca_reducted_embedding_articles

def clicks_essentials_informations(df_clicks: pd.DataFrame):
    """
    Récupérer les informations essentielles des interactions clients-articles

    Parameters:
    df_clicks: la dataframe des interactions complète
     
    Return:
    Une dataframe ne comprenant que les informations essentielles, à savoir :
    - id du client
    - le nombre de clique du client
    - l'id du dernier article consulté (cliqué)
    """

    df_essential_informations = pd.DataFrame()
    df_essential_informations["nb_clicks"] = df_clicks.groupby("user_id").size()
    last_article = df_clicks.loc[df_clicks.groupby("user_id")["click_timestamp"].idxmax(), ["user_id", "click_article_id"]]
    last_article = last_article.set_index("user_id")
    df_essential_informations["last_article"] = last_article["click_article_id"]

    return df_essential_informations

def _cosine_similar_articles(article_id, articles_embeddings):
    """Trouver les articles les plus similaires à un article donné avec la cosinus similarité

    Parameters:
    article_id: l'identifiant de l'article
    articles_embeddings: les embeddings des articles
    """

    # récupération du dernier article
    last_article = articles_embeddings.loc[article_id]

    # Convertir en tableau numpy pour un calcul rapide
    matrix = articles_embeddings.values

    # Normaliser les vecteurs pour éviter les divisions répétées
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_normalized = matrix / norms  # Normalisation ligne par ligne
    last_article_normalized = last_article / np.linalg.norm(last_article)

    # Produit scalaire entre l'article cible et tous les autres
    similarities = np.dot(matrix_normalized, last_article_normalized.T).flatten()

    # Associer les scores aux articles
    return zip(articles_embeddings.index, similarities)

def _nb_clicks_articles_per_user(df_clicks: pd.DataFrame):
    """Obtenir le nombre de clics des articles par utilisateur
    """

    # Nombre de clics par utilisateur
    nb_click_article_per_user = pd.DataFrame(df_clicks.groupby(["user_id", "click_article_id"]).size())

    return nb_click_article_per_user

def _pred_svds(clicks: pd.DataFrame, n_factors:int = 5):
    """Prédire les clics des utilisateurs avec la factorisation de matrice SVD
    
    Parameters:
    clicks: la dataframe contenant les interactions utilisateur-article
    min_clicks: le nombre minimal de clics par utilisateur
    n_factors: le nombre de facteurs
    """

    # Obtenir le nombre de clics par utilisateur
    clicks_articles_per_user_df = _nb_clicks_articles_per_user(clicks)

    # Créer la matrice utilisateur-article
    clicks_articles_matrix = clicks_articles_per_user_df.pivot_table(index="user_id", columns="click_article_id", values=0, fill_value=0)
    clicks_articles_matrix_sparse = csr_matrix(clicks_articles_matrix.values)
    U, sigma, Vt = svds(clicks_articles_matrix_sparse, k=n_factors)


    # Prédire les clics
    predicted_rating = np.dot(np.dot(U, np.diag(sigma)), Vt)
    predicted_rating_norm = (predicted_rating - predicted_rating.min()) / (predicted_rating.max() - predicted_rating.min())


    predicted_rating_norm_df = pd.DataFrame(predicted_rating_norm, columns=clicks_articles_matrix.columns)
    predicted_rating_norm_df.index = clicks_articles_matrix.index

    return predicted_rating_norm_df

def preds_svds_user(df_clicks: pd.DataFrame, n_factors:int = 15, user_min_clicks:int = 10, article_min_clicks:int = 100):
    """Prédire les clics de l'utilisateurs avec la factorisation de matrice SVD
    
    Parameters:
    user_id: l'identifiant de l'utilisateur 
    clicks: la dataframe contenant les interactions utilisateur-article
    n_factors: le nombre de facteurs
    min_clicks: le nombre minimal de clics par utilisateur
    article_min_clicks: le nombre minimal de clicks pour que l'article soit pris en compte
    """

    # Obtenir les intereactions utilisateur-article à un format réduit
    nb_clicks_articles = df_clicks.groupby("click_article_id").size().sort_values(ascending=False)
    nb_clicks_users = df_clicks.groupby("user_id").size().sort_values(ascending=False)

    filtered_articles  = nb_clicks_articles[nb_clicks_articles >= user_min_clicks]
    filtered_users = nb_clicks_users[nb_clicks_users >= article_min_clicks]
    filtered_clicks = df_clicks[df_clicks["click_article_id"].isin(filtered_articles.index) & df_clicks["user_id"].isin(filtered_users.index)]

    # Obtenir les prédictions svds
    predicted_rating_norm = _pred_svds(filtered_clicks, n_factors)

    return predicted_rating_norm