import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler

############################################################################################################
# Fonctions internes
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

    return last_article["click_article_id"].values.item()

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
    similarities = zip(articles_embeddings.index, similarities)
    similar_articles_series = pd.Series(
        {article_id: score for article_id, score in similarities}
    )

    return similar_articles_series

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
    predicted_rating_norm_df.index = clicks_articles_matrix.index

    return predicted_rating_norm_df

def _best_cosine_similar_articles(article_id, articles_embeddings, n: int):
    """Trouver les 5 articles les plus similaires à un article donné avec la cosinus similarité

    Parameters:
    article_id: l'identifiant de l'article
    articles_embeddings: les embeddings des articles
    n: le nombre d'articles similaires à retourner
    """

    similarities = _cosine_similar_articles(article_id, articles_embeddings)
    if n is not None:
        best_similarities = similarities.sort_values(ascending=False)
        best_similarities = best_similarities.iloc[1:n+1]
    else:
        best_similarities = similarities

    return best_similarities

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
    article_id = _last_clicked_articles_user(user_id, clicks)
    similar_articles = _cosine_similar_articles(article_id, articles_embeddings)

    return similar_articles

def best_cosine_similar_articles_with_article_id(article_id: int, articles_embeddings: pd.DataFrame, n: int = 5):
    """Obtenir les n articles les plus similaires pour l'utilisateur donné
    
    Parameters:
    article_id: l'article de référence
    articles_embeddings: les embeddings des articles
    n: le nombre d'articles similaires
    """
    
    # Obtenir les n articles les plus similaires pour chaque utilisateur
    best_similar_articles = _best_cosine_similar_articles(article_id, articles_embeddings, n)

    return best_similar_articles

def best_cosine_similar_articles_per_user(user_id: int, clicks: pd.DataFrame, articles_embeddings: pd.DataFrame, n: int = 5):
    """Obtenir les n articles les plus similaires pour l'utilisateur donné
    
    Parameters:
    user_id: l'identifiant de l'utilisateur
    clicks: la dataframe contenant les clics
    articles_embeddings: les embeddings des articles
    n: le nombre d'articles similaires
    """
    
    # Obtenir les n articles les plus similaires pour chaque utilisateur
    article_id = _last_clicked_articles_user(user_id, clicks)
    best_similar_articles_series = best_cosine_similar_articles_with_article_id(article_id, articles_embeddings, n)

    return best_similar_articles_series

def preds_svds_user(user_id: int, clicks: pd.DataFrame, n_factors:int = 15, user_min_clicks:int = 10, article_min_clicks:int = 100):
    """Prédire les clics de l'utilisateurs avec la factorisation de matrice SVD
    
    Parameters:
    user_id: l'identifiant de l'utilisateur 
    clicks: la dataframe contenant les interactions utilisateur-article
    n_factors: le nombre de facteurs
    min_clicks: le nombre minimal de clics par utilisateur
    article_min_clicks: le nombre minimal de clicks pour que l'article soit pris en compte
    """

    # Obtenir les intereactions utilisateur-article à un format réduit
    nb_clicks_articles = clicks.groupby("click_article_id").size().sort_values(ascending=False)
    nb_clicks_users = clicks.groupby("user_id").size().sort_values(ascending=False)

    filtered_articles  = nb_clicks_articles[nb_clicks_articles >= user_min_clicks]
    filtered_users = nb_clicks_users[nb_clicks_users >= article_min_clicks]
    filtered_clicks = clicks[clicks["click_article_id"].isin(filtered_articles.index) & clicks["user_id"].isin(filtered_users.index)]

    # Obtenir les prédictions svds
    if user_id in filtered_clicks["user_id"].values:
        predicted_rating_norm = _pred_svds(filtered_clicks, n_factors)
        predicted_rating_norm_user = predicted_rating_norm.loc[user_id]
    else:
        predicted_rating_norm_user = None

    return predicted_rating_norm_user

def best_preds_svds_user(user_id: int, clicks: pd.DataFrame, n_factors:int = 15, user_min_clicks:int = 10, article_min_clicks:int = 100, n:int = 5):
    """Obtenir les meilleurs prédictions pour un utilisateur donné
    
    Parameters:
    user_id: l'identifiant de l'utilisateur
    clicks: la dataframe contenant les interactions utilisateur-article
    n_factors: le nombre de facteurs
    min_clicks: le nombre minimal de clics par utilisateur
    article_min_clicks: le nombre minimal de clicks pour que l'article soit pris en compte
    """

    # Obtenir les n articles recommandés avec la factorisation de matrice SVD
    collaborative_filtering_articles = preds_svds_user(user_id, clicks, n_factors, user_min_clicks, article_min_clicks)
    best_articles = collaborative_filtering_articles.sort_values(ascending=False)[:n]

    # Obtenir les meilleures prédictions pour l'utilisateur
    return best_articles

def hybrid_recommender(user_id: int, clicks: pd.DataFrame, articles_embeddings: pd.DataFrame, n_factors:int = 15, n: int = 5, user_min_clicks: int = 10, article_min_clicks: int = 270):
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

    user_id = int(user_id)
    user_clicks = clicks.loc[clicks["user_id"] == user_id].groupby("click_article_id").size().sum()

    if user_clicks == 0:
        return "most popular", most_popular_articles(clicks, n)
    elif user_clicks < user_min_clicks:
        return "best cosine similar", best_cosine_similar_articles_per_user(user_id, clicks, articles_embeddings, n)
    else:

        # Obtenir les n articles les plus similaires pour l'utilisateur (Content Based)
        content_based_articles = cosine_similar_articles_per_user(user_id, clicks, articles_embeddings)
        # Obtenir les n articles recommandés avec la factorisation de matrice SVD
        collaborative_filtering_articles = preds_svds_user(user_id, clicks, n_factors)

        # Si l'utilisateur est bien concerné dans collaborative filtering
        if collaborative_filtering_articles is not None:
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
        else:
            return "best cosine similar", content_based_articles.sort_values(ascending=False)[1:n+1]

def fast_hybrid_recommender(user_id: int, df_most_popular:pd.DataFrame, df_clicks_essential_informations: pd.DataFrame, df_svds: pd.DataFrame, df_articles_embeddings: pd.DataFrame, n:int = 5):
    """
    Version de système de recommandations plus rapide en se basant sur des datasets précalculés au lieu d'effectuer des calculs à la volée

    Parameters:
    user_id: l'identifiant de l'utilisateur
    df_most_popular: DataFrame des articles les plus populaires
    df_clicks_essential_informations: DataFrame contenant les informations essentielles des interactions utilisateurs-articles
    df_svds: DataFrame contenant les prédictions SVDS précalculés
    df_articles_embeddings: DataFrame contenant les embeddings des articles
    n: le nombre d'articles à recommander
    """

    unknow_user = False
    # Si l'utilisateur existe
    if user_id in df_clicks_essential_informations.index:
        # Obtenir le nombre de cliques de l'utilisateur
        info_user = df_clicks_essential_informations.loc[user_id]
    else:
        unknow_user = True

    # Si utilisateur inconnu ou nouveau utilisateur
    if unknow_user or info_user["nb_clicks"] == 0:
        return "most_popular", df_most_popular
    else:
        # Calculer le content based filtering en utilisant la similarité cosinus
        content_based_articles = _cosine_similar_articles(info_user["last_article"], df_articles_embeddings)

        # Vérifier si l'utilisateur n'est pas inclus dans les prédictions SVDS
        if user_id not in df_svds.index:
            # Récupérer les articles les plus similiars au dernier article cliqué par l'utilisatgeur
            best_similarities = content_based_articles.sort_values(ascending=False)
            best_similarities = best_similarities[1:n+1]
            return "content based filtering", best_similarities 
        else:
            collaborative_filtering_articles = df_svds.loc[user_id]
            
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