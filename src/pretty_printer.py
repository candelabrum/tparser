class PrettyPrinter:
    def print_best_sentences(self, best_sentences_df):
        """
        print sentences
        """
        best_sentences_df = best_sentences_df.set_index('label_of_cluster').join(best_sentences_df['label_of_cluster'].value_counts())
        best_sentences_df = best_sentences_df.sort_values(by='count', ascending=False)
        best_sentences_df = best_sentences_df.reset_index()
        best_sentences_df = best_sentences_df.query('count > 0')
        for cluster_id in best_sentences_df['label_of_cluster'].drop_duplicates().values.tolist():
            output = best_sentences_df.query("label_of_cluster == @cluster_id")['sent'].values.tolist()
            for sent in output:
                print('-', sent)
            print("====================================================")
        return output
