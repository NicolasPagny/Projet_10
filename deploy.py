############################################################################################################
# Ce fichier contient des fonctions de créations des datasets nécessaires au bon fonctionnement des systèmes
# de recommandations
############################################################################################################

# Importations
import recommender
import prepare_files
import os
import pandas as pd
print("importations des fichiers terminés")

# Paths
DATASETS_PATH = "datasets/"
ARTICLES_EMBEDDINGS_PATH = DATASETS_PATH + "articles_embeddings.pickle"
ARTICLES_METADATA_PATH = DATASETS_PATH + "articles_metadata.csv"
CLICKS_PATH = DATASETS_PATH + "clicks/"
CLICKS_HOUR_CONCATENATED_PATH = DATASETS_PATH + "clicks_hour_concatenated.csv"
PCA_REDUCTED_ARTICLES_EMBEDDINGS_PATH = DATASETS_PATH + "pca_reducted_articles_embeddings.pickle"
SVDS_PATH = DATASETS_PATH + "svds.pickle"
CLICKS_ESSENTIALS_INFORMATIONS_PATH = DATASETS_PATH + "clicks_essentials_informations.pickle"
MOST_POPULAR_ARTICLES_PATH = DATASETS_PATH + "most_popular_articles.pickle"
print("initialisation des paths terminée")

# Créer le fichier concatené des datasets clicks
df_clicks_hour = pd.DataFrame()

clicks_hour_files = os.listdir(CLICKS_PATH)
for file in clicks_hour_files:
    df_clicks_hour = pd.concat([df_clicks_hour, pd.read_csv(CLICKS_PATH + file)], ignore_index=True)

df_clicks_hour.to_csv(CLICKS_HOUR_CONCATENATED_PATH, index=False)
print("Création de clicks_hour terminée")

# Créer le dataset most_popular
df_most_popular = prepare_files.most_popular_articles(df_clicks_hour, 5)
df_most_popular.to_pickle(MOST_POPULAR_ARTICLES_PATH)
print("Création du dataset most popular terminé")

# Créer le dataset des embeddings des articles réduits par PCA
df_embedding_articles = pd.DataFrame(pd.read_pickle(ARTICLES_EMBEDDINGS_PATH))
df_pca_reducted_articles_embeddings = prepare_files.pca_reduct_articles_embeddings(df_embedding_articles, df_clicks_hour)
df_pca_reducted_articles_embeddings.to_pickle(PCA_REDUCTED_ARTICLES_EMBEDDINGS_PATH)
print("Création du dataset pca_reducted_article_embeddings terminée")

# Créer le fichier clicks_essentials_informations
df_clicks_essentials_informations = prepare_files.clicks_essentials_informations(df_clicks_hour)
df_clicks_essentials_informations.to_pickle(CLICKS_ESSENTIALS_INFORMATIONS_PATH)
print("Création du dataset clicks_essentials_informations terminée")

# Créer le fichier svds précalculé
df_svds = prepare_files.preds_svds_user(df_clicks_hour)
df_svds.to_pickle(SVDS_PATH)
print("Création du dataset svds terminée")

#Test du bon fonctionnement du déploiement
try:
    df_test_most_popular = pd.read_pickle(MOST_POPULAR_ARTICLES_PATH)
    df_test_pca_reducted_articles_embeddings = pd.read_pickle(PCA_REDUCTED_ARTICLES_EMBEDDINGS_PATH)
    df_test_clicks_essentials_informations = pd.read_pickle(CLICKS_ESSENTIALS_INFORMATIONS_PATH)
    df_test_svds = pd.read_pickle(SVDS_PATH)
    recommender.fast_hybrid_recommender(
        24, 
        df_test_most_popular, 
        df_test_clicks_essentials_informations, 
        df_test_svds, 
        df_test_pca_reducted_articles_embeddings,
        5)
except Exception as e:
    print(f"Erreur du déploiement, erreur : {e}")
print("Test terminé, aucune erreur constatée, déploiement réussi")

