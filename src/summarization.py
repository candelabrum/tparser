import umap
import plotly.express as px
import numpy as np
import pandas as pd
import plotly
from clusterizator import Clusterizator
from utils import filtration_by_regex, get_all_sentences, get_indices
from sentence_transformers import util
from embedder import LaBSEEmbedder
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm


class Summartization:
    def __init__(self, embedder=LaBSEEmbedder()):
        self.embedder = embedder

    def make_summary(self, all_docs_df):
        """
        Arguments: 
        ============================
        all_docs_df - pd.DataFrame['id', 'channel', 'date', 'true_id', 'message', 'embed'] - docs to summary
        where 
        id - is message id in channel
        true_id - is unique id for news.
        =============================
        Constraints:
        all_docs_df need to contain >= 2 news
        =============================
        Returns:
        Summary - pd.DataFrame
        """

        assert all_docs_df.shape[0] > 1

        all_docs_df = self.make_clusterization_by_all_docs(
            all_docs_df, show=False)
        self.show_visualization_of_clusterization(
            all_docs_df, 'clusterization_news.html')
#        print(all_docs_df)
        all_docs_df = all_docs_df.set_index('label').join(
            all_docs_df['label'].value_counts())
        all_docs_df = all_docs_df.sort_values(by='count', ascending=False)
        all_docs_df = all_docs_df.reset_index()
        best_sentences_df, all_filtrated_sents_df = self.make_clusterization_by_sentences(
            all_docs_df, top_k=5)

        return best_sentences_df, all_filtrated_sents_df

    def show_visualization_of_clusterization(self, all_docs_df, path_to_save, show=False):
        """
        Makes html_doc with UMAP visualization of clustering

        Arguments
        =================
        - all_docs_df - pd.DataFrame['id', 'channel', 'date', 'true_id', 'message', 'embed', 'label] - docs to summary
                where 
                id - is message id in channel
                true_id - is unique id for news.
        - path_to_save - str - file to save html file
        - show - bool - show clusterization here or not.
        =============================
        Constraints:
        all_docs_df need to contain >= 2 news
        ============================
        Returns:
        plotly.express.figure of visualization
        """
        umap_error = False
        embeds = all_docs_df['embed'].apply(pd.Series).values
        reducer = umap.UMAP(random_state=139892)

        try:
            embeddings = reducer.fit_transform(embeds)
        except:
            umap_error = True
        if umap_error is False:
            fig = px.scatter(
                embeddings,
                size_max=20,
                size=[0.8] * embeddings.shape[0],
                x=0,
                y=1,
                color=all_docs_df['label'],  # Shorten the legend text
                hover_name=all_docs_df['message'].apply(
                    lambda x: x[:70] + '...')
            )
            if show:
                fig.show()
                plotly.offline.plot(fig, filename=path_to_save)

    def make_clusterization_by_all_docs(self, all_docs_df, show=False):
        """
        Makes clusterization by news. Add new column to all_docs_df

        Arguments
        =================
        all_docs_df - pd.DataFrame['id', 'channel', 'date', 'true_id', 'message', 'embed'] - docs to summary
        where 
        id - is message id in channel
        true_id - is unique id for news.
        =============================
        Constraints:
        all_docs_df need to contain >= 2 news
        ============================
        Returns:
        all_docs_df - pd.DataFrame['id', 'channel', 'date', 'true_id', 'message', 'embed', 'label'] 
        """

        embeds = all_docs_df['embed'].apply(pd.Series).values
        matrix_corr = util.dot_score(embeds, embeds)
        cluster = AgglomerativeClustering(
            linkage='average',
            n_clusters=None,
            metric='precomputed',
            distance_threshold=0.27
        )
        cluster.fit(1 - matrix_corr)
        all_docs_df['label'] = cluster.labels_

        return all_docs_df

    def make_filtration_sentences(self, all_sents_df):
        """
        all_sents_df

        """
        embeds = all_sents_df['embed'].apply(pd.Series).values
#        if show:
#            show_clusterization_sentence(all_sents_df)
        corr_matrix = util.dot_score(embeds, embeds)
        dct = dict()
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[0]):
                if i > j:
                    dct[(i, j)] = float(corr_matrix[i, j])

        series = pd.Series(dct).sort_values(ascending=False)
        pairs = series[series > 0.99].index.tolist()
        if len(pairs) > 0:
            list_to_del = pd.DataFrame(pairs)[0].values
            list_to_del = list(set(list_to_del))
        else:
            list_to_del = []
        all_sents_df = all_sents_df.reset_index()
        all_sents_df = all_sents_df[~all_sents_df['index'].isin(list_to_del)]
        all_sents_df['number_of_pronouns'] = all_sents_df['sent'].apply(
            filtration_by_regex)
        all_sents_df = all_sents_df.query('number_of_pronouns <= 1')
        all_sents_df['blacklist'] = all_sents_df['sent'].apply(
            lambda x: 'Подписаться' in x or 'Подписывайтесь' in x or 'Mash' in x or '@kommersant' in x or len(
                x) < 25
        )
        all_sents_df = all_sents_df.query("blacklist == False")
        all_sents_df = all_sents_df.sort_values(by='date')
        all_senttences_selected2 = []
        for row_idx, row in all_sents_df.iterrows():
            all_senttences_selected2.append({
                'true_id': row['true_id'],
                'sent_idx': row['sent_idx'],
                'sent': row['sent'],
                'embed': row['embed'],
                'date': row['date']
            })

        return all_senttences_selected2

    def make_clusterization_by_sentences(self, all_docs_df, top_k=5, show=False):
        """
        Makes clusterization by sentences in every cluster.

        Arguments
        =================
        all_docs_df - pd.DataFrame['id', 'channel', 'date', 'true_id', 'message', 'embed', 'label'] - docs to summary
        where 
        id - is message id in channel
        true_id - is unique id for news.
        =============================
        Constraints:
        all_docs_df need to contain >= 2 news
        ============================
        Returns:
        - best_sentences_df
        - clusters_with_all_filtrated_sentences
        """
        best_sentences_list = []
        filtrated_sentences_list = []
        for label_ in tqdm(all_docs_df['label'].value_counts().index.tolist()):
            all_docs_df_selected = all_docs_df.query("label == @label_")
            all_sentences_selected = get_all_sentences(all_docs_df_selected)
            all_sents_df = pd.DataFrame(all_sentences_selected)
            if show:
                self.show_visualization_of_clusterization(all_sents_df)
            all_sentences_selected2 = self.make_filtration_sentences(
                all_sents_df)
            if len(all_sentences_selected2) > 1:
                printed_clusters, labels_in_cluster = self.get_clusters(
                    all_sentences_selected2)
                all_senttences_selected2_df = pd.DataFrame(
                    all_sentences_selected2
                ).assign(
                    labels_in_cluster=labels_in_cluster
                ).assign(
                    label_of_cluster=label_
                )
                best_sentences_df = pd.DataFrame(
                    printed_clusters,
                    columns=['true_id', 'sent', 'sent_idx', 'cluster_idx']
                ).assign(label_of_cluster=label_)

                best_sentences_list.append(best_sentences_df)
                filtrated_sentences_list.append(all_senttences_selected2_df)

            if len(best_sentences_list) > top_k:
                break
        #    break;
        return pd.concat(best_sentences_list), pd.concat(filtrated_sentences_list)

    def get_clusters(self, all_sents, top_k=10000):
        """
        Performs clusterization and selection sentence from cluster

        Arguments:
        ===============
        all_sents - list[Sentence]
        ==============
        Returns 

        """

        clusterizator = Clusterizator()
        labels = clusterizator.get_labels(all_sents)
        indices = [index[0]
                   for index in pd.DataFrame(labels).value_counts().index]
        selected_clusters = []
        for index in indices:
            if len(selected_clusters) > top_k:
                break
            selected_indices = get_indices(labels, index)
            embeddings = []
            mean_position_lst = []
            for sel_index in selected_indices:
                embeddings.append(all_sents[sel_index]['embed'])
                mean_position_lst.append(all_sents[sel_index]['sent_idx'])

            mean_position = np.mean(mean_position_lst)
            selected_clusters.append(
                (index, mean_position, len(selected_indices)))
            continue

    #    print('selected_clusters: ', selected_clusters)
        best_sentences = clusterizator.get_best_sentences(
            all_sents, selected_clusters)

        return best_sentences, labels
