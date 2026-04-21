from nltk.corpus.reader import CategorizedPlaintextCorpusReader

def create_corpus_reader(dataset_path):
    return CategorizedPlaintextCorpusReader(dataset_path, r'.*\.txt', cat_pattern=r'(\w+)/.*')

# Example usage:
# movie_reviews = load_movie_reviews('path/to/your/dataset/')
