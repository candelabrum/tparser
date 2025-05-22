import nltk
import re
import umap
import plotly.express as px
import numpy as np
import pandas as pd
import plotly
from embedder import LaBSEEmbedder
from sentence_transformers import util
from sklearn.cluster import AgglomerativeClustering
from whoosh.qparser import QueryParser
from tqdm import tqdm
from whoosh.index import open_dir
from dateutil import relativedelta
from whoosh.query import DateRange


# Function to retrieve all documents from a Whoosh index
def get_all_documents(index_dir, filt, date):
    try:
        # Open the Whoosh index directory
        index = open_dir(index_dir)
        print('index.doc_count', index.doc_count())
        # Create a searcher object
        with index.searcher() as searcher:
            # Query to match all documents
            if filt is None:
                query = QueryParser("message", index.schema).parse('*')
            else:
                query = QueryParser("message", index.schema).parse(f'*{filt}*')
            if date is not None:
                query = DateRange(
                    "date",
                    date,
                    date + relativedelta.relativedelta(days=1)
                ) & query
            results = searcher.search(query, limit=index.doc_count())
            # Collecting all documents
            documents = [dict(result) for result in results]
            return documents
    except Exception as e:
        return f"An error occurred: {e}"


def get_cos(first_kvart, second_kvart):
    denominator = np.linalg.norm(
        first_kvart['embed']
    ) * np.linalg.norm(
        second_kvart['embed']
    )

    return np.dot(first_kvart['embed'], second_kvart['embed']) / denominator


def get_all_sents(index_name, embedder, filt=None, date=None):
    all_docs = get_all_documents(index_name, filt, date)
    all_sents = []
    print(all_docs)
    for doc in tqdm(all_docs):
        if filt is not None and filt in doc['message']:
            true_id = doc['true_id']
            sents_text = nltk.sent_tokenize(doc['message'])

            for sent_idx, sent in enumerate(sents_text):
                #                print(sent)
                all_sents.append({
                    'true_id': true_id,
                    'sent_idx': sent_idx,
                    'sent': sent,
                    'embed': embedder.get_embed(sent),
                    'date': doc['date']
                })

    return all_sents, all_docs


def get_indices(labels, index):
    return [label_idx for label_idx, label in enumerate(labels) if label == index]


def filtration_by_regex(text):
    pronouns_pattern = r'\b(я|ты|он|она|оно|мы|вы|они|меня|тебя|его|её|ему|ей|нас|вас|их|ими|мой|твой|его|её|наш|ваш|их|кто|что|какой|который|чей|где|как|когда|зачем|почему|всё|вся|всё|каждый|всякий|любой|некоторые|другие|другой|сам|самый|эта|это|эту|эти|этой)\b'
    return len(re.findall(pronouns_pattern, text, re.IGNORECASE))


def show_clusterization_by_all_docs(all_docs, show=False, embedder=LaBSEEmbedder()):
    # get all_docs_df
    print('clustering by all docs')
#    true_id_str_to_int = fts.counter.true_id_str_to_int# true_ids
#    int_to_true_id_str = {v: k for k, v in true_id_str_to_int.items()}
#    true_id2text = {doc['true_id']:doc['message'] for doc in all_docs}
    for doc in tqdm(all_docs):
        doc['embed'] = embedder.get_embed(doc['message'])

    all_docs_df = pd.DataFrame(all_docs)

    # clusterization is going here...
    all_docs_df_selected = all_docs_df  # .query('label != -1')
    embeds = all_docs_df_selected['embed'].apply(pd.Series).values
    reducer = umap.UMAP(random_state=139892)
    print('embeds.shape', embeds.shape)
    umap_error = False
    try:
        embeddings = reducer.fit_transform(embeds)
    except Exception:
        umap_error = True
    matrix_corr = util.dot_score(embeds, embeds)
    cluster = AgglomerativeClustering(
        linkage='average',
        n_clusters=None,
        metric='precomputed',
        distance_threshold=0.27
    )  # , connectivity=matrix_corr, n_clusters=10)

    cluster.fit(1 - matrix_corr)
    labels = cluster.labels_
    all_docs_df['label'] = labels

    # plot predictions
    if umap_error is False:
        fig = px.scatter(
            embeddings,
            size_max=20,
            size=[0.8] * embeddings.shape[0],
            x=0,
            y=1,
            color=all_docs_df_selected['label'],  # Shorten the legend text
            hover_name=all_docs_df_selected['message'].apply(
                lambda x: x[:70] + '...'),  # Show full text on hover
            #   labels={'type_of_cluster': 'Shortened Message'}
        )
        if show:
            fig.show()
            plotly.offline.plot(fig, filename='news_clusterization.html')

    return all_docs_df


def get_all_sentences(all_docs_df, embedder=LaBSEEmbedder()):
    top_k = 1000
    top_clusters = all_docs_df['label'].value_counts().index.tolist()[:top_k]

    for top_cluster in tqdm(top_clusters):
        all_sents = []
        all_docs_selected = all_docs_df.query("label == @top_cluster")
        for doc_index, doc in all_docs_selected.iterrows():
            for sent_idx, sent in enumerate(nltk.sent_tokenize(doc['message'])):
                all_sents.append({
                    'true_id': doc['true_id'],
                    'sent_idx': sent_idx,
                    'sent': sent,
                    'embed': embedder.get_embed(sent),
                    'date': str(doc['date'])
                })

    return all_sents
