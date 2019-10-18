import tensorflow as tf
import numpy as np
from DataLoader import DataLoader
from Embedding_Layer import EmbeddingLayer


def main():
    data = DataLoader('C:/Users/user/Desktop/R&E/datasets/PWKP_108016/temp.txt')
    normal, simple = data.load_data(data)
    embedding = EmbeddingLayer('C:/Users/user/Desktop/R&E/pretrained_glove/glove.6B.50d.txt', 50, normal, simple)

    x_train, x_test, y_train, y_test = data.split_data(normal, simple)

    for line in normal:
        line = embedding.temp_embed([line])
        for char in line[0]:
            char = embedding.embed([char])[0]


if __name__ == "__main__":
    main()
