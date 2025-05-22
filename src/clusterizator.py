import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import get_cos
from sklearn.cluster import AgglomerativeClustering


class Clusterizator:
    def compute_sim_matrix(self, all_sents):
        matrix_corr = np.zeros((len(all_sents), len(all_sents)))
        for first_idx, first_kvart in tqdm(enumerate(all_sents)):
            for second_idx, second_kvart in enumerate(all_sents):
                if first_kvart['true_id'] != second_kvart['true_id']:
                    matrix_corr[first_idx, second_idx] = get_cos(
                        first_kvart, second_kvart)

                elif first_kvart['true_id'] == second_kvart['true_id'] and first_kvart['sent_idx'] == second_kvart['sent_idx']:
                    matrix_corr[first_idx, second_idx] = 1
                elif first_kvart['true_id'] == second_kvart['true_id'] and first_kvart['sent_idx'] != second_kvart['sent_idx']:
                    matrix_corr[first_idx, second_idx] = 0
                else:
                    assert "Never has been"

        self.matrix_corr = matrix_corr

        return matrix_corr

    def get_labels(self, all_sents, verbose=False):
        """
        Perform agglomerative clustering for sentences

        ======================
        Returns cluster labels
        list[int]
        """
        matrix_corr = self.compute_sim_matrix(all_sents)
        cluster = AgglomerativeClustering(
            linkage='average',
            n_clusters=None,
            metric='precomputed',
            distance_threshold=0.27
        )  # , connectivity=matrix_corr, n_clusters=10)

        cluster.fit(1 - matrix_corr)
        self.labels_ = cluster.labels_

        return cluster.labels_

    def reorder_clusters(self, clusters_info):
        """
        Reorder clusters by average sentence position in doc

        Arguments:
        ===========================
        - clusters_info - list[Tuple['cluster_id', 'mean_position', 'count']]
        ===========================
        Returns reordered clusters_info
        clusters_info - pd.DataFrame['cluster_id', 'mean_position', 'count']
        """
        clusters_info = pd.DataFrame(
            clusters_info,
            columns=['cluster_id', 'mean_position', 'count']
        ).sort_values(by='mean_position').query("count > 2")['cluster_id'].values.tolist()

        return clusters_info

    def select_best_sentence_from_cluster(self, embeddings_of_sents):
        """
        Get index of best sentence by embeddings

        Arguments:
        ===========================
        - embeddings_of_sents - list[np.array]
        ===========================
        Returns index of best sentence
        Int
        """

        return np.argmin(
            [
                np.abs(np.mean(embeddings_of_sents, axis=0) -
                       np.array(embeddings_of_sents)).mean()
                for emb in embeddings_of_sents
            ]
        )

    def get_best_sentences(self, all_sents, cluster_indices):
        """
        Reorder clusters and select best sentence from cluster

        Arguments:
        ===================
        - all_sents -
        - cluster_indices - list[Tuple['cluster_id', 'mean_position', 'count']]
        ===================
        Returns list of best sentences in correct order to print
        list[Sentence]

        """
        output = []
        clusters_order = self.reorder_clusters(cluster_indices)
        all_sents_with_cluster = list(zip(all_sents, self.labels_))
        for cluster_to_print in clusters_order:
            sentences = []
            embeddings = []
            for sent, cluster_idx in all_sents_with_cluster:
                if cluster_idx == cluster_to_print:
                    sentences.append(sent)
                    embeddings.append(sent['embed'])
            best_sentence_index = self.select_best_sentence_from_cluster(
                embeddings)
            sent = sentences[best_sentence_index]
            output.append((sent['true_id'], sent['sent'],
                          sent['sent_idx'], cluster_idx))

        return output
