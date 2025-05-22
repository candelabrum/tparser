from embedder import LaBSEEmbedder
from summarization import Summartization
from pretty_printer import PrettyPrinter


def get_summary(all_docs_df, calculate_embeds=True):
    """
    all_docs_df - pd.DataFrame[id, 	channel, 	date, 	true_id, 	message]

    """
    print("I am starting calculate embeddings, time required ~", all_docs_df.shape[0] * 0.5, "seconds")
    if calculate_embeds:
        all_docs_df['embed'] = all_docs_df['message'].apply(
            lambda x: LaBSEEmbedder().get_embed(x))
    summarizator = Summartization().make_summary(all_docs_df)
    tmp_df = summarizator[0]
    tmp_df = tmp_df.set_index('label_of_cluster').join(
        tmp_df['label_of_cluster'].value_counts())
    tmp_df = tmp_df.sort_values(by='count', ascending=False)
    tmp_df = tmp_df.reset_index()
    _ = PrettyPrinter().print_best_sentences(tmp_df.drop('count', axis=1))
