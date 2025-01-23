import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler

############################################################################################################
# Fonctions internes (non utilisées avec azure fonction)
############################################################################################################

def _cosine_similarity(vector1, vector2):
    """Calculer la similarité cosinus entre les articles
    """

    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return similarity

def _clicked_articles_user(user_id: int, clicks: pd.DataFrame):
    """obtenir les articles cliqués par un utilisateur donné
    
    Parameters:
    user_id: l'identifiant de l'utilisateur
    clicks: la dataframe contenant les clics
    """

    # Sélectionner les articles cliqués par l'utilisateur
    articles_user = clicks[clicks["user_id"] == user_id]
    articles_user = articles_user[["click_article_id", "click_timestamp"]]

    return articles_user

def _nb_clicks_per_article(clicks: pd.DataFrame):
    """Obtenir le nombre de clics par article
    
    Parameters:
    clicks: la dataframe contenant les clics
    """

    # Compter le nombre de clics par article
    nb_clicks = clicks["click_article_id"].value_counts()

    return nb_clicks

def _last_clicked_articles_user(user_id: int, clicks: pd.DataFrame):
    """Obtenir le dernier article cliqué par un utilisateur

    Parameters:
    user_id: l'identifiant de l'utilisateur
    clicks: la dataframe contenant les clics
    """

    articles_user = _clicked_articles_user(user_id, clicks)
    last_article = articles_user[articles_user["click_timestamp"] == articles_user["click_timestamp"].max()]

    return last_article

def _cosine_similar_articles(article_id, articles_embeddings):
    """Trouver les articles les plus similaires à un article donné avec la cosinus similarité

    Parameters:
    article_id: l'identifiant de l'article
    articles_embeddings: les embeddings des articles
    """

    article_embedding = articles_embeddings[article_id].ravel()
    similarities = {}
    for id, embedding in enumerate(articles_embeddings):
        similarities[id] = _cosine_similarity(article_embedding, embedding)
    return similarities.items()

def _best_cosine_similar_articles(article_id, articles_embeddings, n: int = None):
    """Trouver les 5 articles les plus similaires à un article donné avec la cosinus similarité

    Parameters:
    article_id: l'identifiant de l'article
    articles_embeddings: les embeddings des articles
    n: le nombre d'articles similaires à retourner
    """

    similarities = _cosine_similar_articles(article_id, articles_embeddings)
    if n is not None:
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        best_similarities = similarities[1:n+1]
    else:
        best_similarities = similarities
    return best_similarities

def _nb_clicks_articles_per_user(clicks: pd.DataFrame):
    """Obtenir le nombre de clics des articles par utilisateur
    """

    # Nombre de clics par utilisateur
    nb_click_article_per_user = pd.DataFrame(clicks.groupby(["user_id", "click_article_id"]).size())

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
    predicted_rating_norm_df.index.name = "user_id"

    return predicted_rating_norm_df

############################################################################################################
# Fonctions utilisées dans azure fonctions
############################################################################################################

def most_popular_articles(clicks: pd.DataFrame, n: int = 5):
    """Obtenir les articles les plus populaires
    
    Parameters:
    clicks: la dataframe contenant les clics
    n: le nombre d'articles à retourner
    """

    # Obtenir le nombre de clics par article
    nb_clicks = _nb_clicks_per_article(clicks)
    nb_clicks = nb_clicks.sort_values(ascending=False)

    # Sélectionner les n articles les plus populaires
    most_popular_articles = nb_clicks.head(n)

    return most_popular_articles

def cosine_similar_articles_per_user(user_id:int, clicks: pd.DataFrame, articles_embeddings: pd.DataFrame):
    """Obtenir les articles les plus similaires pour un utilisateur donné
    
    Parameters:
    user_id: l'identifiants de l'utilisateur
    clicks: la dataframe contenant les clics
    articles_embeddings: les embeddings des articles
    """
    
    # Obtenir les articles les plus similaires pour l'utilisateur
    article_id = _last_clicked_articles_user(user_id, clicks)["click_article_id"]
    similar_articles = _cosine_similar_articles(article_id, articles_embeddings)
    similar_articles_series = pd.Series(
        {article_id: score for article_id, score in similar_articles}
    )

    return similar_articles_series

def best_cosine_similar_articles_per_user(user_id: int, clicks: pd.DataFrame, articles_embeddings: pd.DataFrame, n: int = 5):
    """Obtenir les n articles les plus similaires pour l'utilisateur donné
    
    Parameters:
    user_id: l'identifiant de l'utilisateur
    clicks: la dataframe contenant les clics
    articles_embeddings: les embeddings des articles
    n: le nombre d'articles similaires
    """

    # Obtenir les n articles les plus similaires pour chaque utilisateur
    article_id = _last_clicked_articles_user(user_id, clicks)["click_article_id"]
    similar_articles = _best_cosine_similar_articles(article_id, articles_embeddings, n)
    best_similar_articles_series = pd.Series(
        {article_id: score for article_id, score in similar_articles}
    )

    return best_similar_articles_series

def preds_svds_user(user_id: int, clicks: pd.DataFrame, n_factors:int = 5):
    """Prédire les clics de l'utilisateurs avec la factorisation de matrice SVD
    
    Parameters:
    user_id: l'identifiant de l'utilisateur 
    clicks: la dataframe contenant les interactions utilisateur-article
    min_clicks: le nombre minimal de clics par utilisateur
    n_factors: le nombre de facteurs
    """

    # Obtenir les prédictions svds
    predicted_rating_norm = _pred_svds(clicks, n_factors)
    predicted_rating_norm_user = predicted_rating_norm.loc[user_id]

    return predicted_rating_norm_user

def best_preds_svds_user(user_id: int, clicks: pd.DataFrame, min_clicks: int = 2, n_factors:int = 5, n: int = 5):
    """Obtenir les meilleurs prédictions pour un utilisateur donné
    
    Parameters:
    user_id: l'identifiant de l'utilisateur
    clicks: la dataframe contenant les interactions utilisateur-article
    predicted_rating_norm: les prédictions des clics
    """

    # Obtenir les prédictions svds
    predicted_rating_norm = _pred_svds(clicks, min_clicks, n_factors)

    # Obtenir les meilleures prédictions pour l'utilisateur
    return predicted_rating_norm.loc[user_id].nlargest(n)

def hybrid_recommander(user_id: int, clicks: pd.DataFrame, articles_embeddings: pd.DataFrame, n_factors:int = 15, n: int = 5, user_min_clicks: int = 10, article_min_clicks: int = 270):
    """Recommander des articles à un utilisateur en utilisant une approche hybride
    
    Parameters:
    user_id: l'identifiant de l'utilisateur
    clicks: la dataframe contenant les clics
    articles_embeddings: les embeddings des articles
    min_clicks: le nombre minimal de clics par utilisateur
    n: le nombre d'articles à recommander
    user_min_clicks: le nombre minimal de clics par utilisateur
    article_min_clicks: le nombre minimal de clics par article
    """

    user_clicks = clicks.loc[clicks["user_id"] == user_id].groupby("click_article_id").size().sum()

    if user_clicks == 0:
        return "most popular", most_popular_articles(clicks, n)
    elif user_clicks < user_min_clicks:
        return "best cosine similar", best_cosine_similar_articles_per_user(user_id, clicks, articles_embeddings, n)
    else:
        # Obtenir les n articles les plus similaires pour chaque utilisateur (Content Based)
        content_based_articles = cosine_similar_articles_per_user(user_id, clicks, articles_embeddings)

        # Obtenir les intereactions utilisateur-article à un format réduit
        nb_clicks_articles = clicks.groupby("click_article_id").size().sort_values(ascending=False)
        nb_clicks_users = clicks.groupby("user_id").size().sort_values(ascending=False)

        filtered_articles  = nb_clicks_articles[nb_clicks_articles >= user_min_clicks]
        filtered_users = nb_clicks_users[nb_clicks_users >= article_min_clicks]
        filtered_clicks = clicks[clicks["click_article_id"].isin(filtered_articles.index) & clicks["user_id"].isin(filtered_users.index)]

        # Obtenir les n articles recommandés avec la factorisation de matrice SVD
        collaborative_filtering_articles = preds_svds_user(user_id, filtered_clicks, n_factors)

        # Garder les articles correspondant aux articles filtrés
        index_to_keep = list(collaborative_filtering_articles.keys())
        filtered_content_based_articles = pd.Series({key: value for key, value in content_based_articles.items() if key in index_to_keep})
        filtered_content_based_articles.index.name = "click_article_id"

        # Normaliser les deux recommandations afin de les combiner
        scaler = MinMaxScaler()
        cb_scaled = scaler.fit_transform(filtered_content_based_articles.values.reshape(-1, 1))
        cf_scaled = scaler.fit_transform(collaborative_filtering_articles.values.reshape(-1, 1))

        # Combiner les deux recommandations
        hybrid_articles = cb_scaled * cf_scaled
        hybrid_articles = pd.Series(hybrid_articles.ravel(), index=filtered_content_based_articles.index)
        best_hybrid_articles = hybrid_articles.nlargest(n)

        return "hybrid", best_hybrid_articles
