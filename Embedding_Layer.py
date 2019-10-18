import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn


class EmbeddingLayer(object):
    def __init__(self, path, embedding_size, normal, simple):
        self.path = path
        self.embedding_size = embedding_size

        all_word = normal + simple
        max_document_length = max([len(x.split()) for x in all_word])

        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        self.vocab_processor.fit_transform(all_word)
        self.vocab_size = len(self.vocab_processor.vocabulary_)
        self.embedding = self.load_glove_embeddings(self.vocab_processor)

    def load_glove_embeddings(self, vocab):
        embeddings = {}
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                w = values[0]
                vectors = np.asarray(values[1:], dtype='float32')
                embeddings[w] = vectors
        vocab_dict = vocab.vocabulary_._mapping
        embedding_matrix = np.random.uniform(-1, 1, size=(self.vocab_size, self.embedding_size))
        num_loaded = 0
        for w, i in vocab_dict.items():
            v = embeddings.get(w)
            if v is not None and i < self.vocab_size:
                embedding_matrix[i] = v
                num_loaded += 1
        embedding_matrix = embedding_matrix.astype(np.float32)
        return embedding_matrix

    def temp_embed(self, line):
        return np.array(list(self.vocab_processor.transform(line)))

    def embed(self, idx):
        sess = tf.Session()
        return tf.nn.embedding_lookup(self.embedding, idx).eval(session=sess)
