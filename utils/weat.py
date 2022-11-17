import json
import numpy as np


def unit_vector(vec):
    """
    Returns unit vector
    """
    return vec / np.linalg.norm(vec)


def cos_sim(v1, v2):
    """
    Returns cosine of the angle between two vectors
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.tensordot(v1_u, v2_u, axes=(-1, -1)), -1.0, 1.0)


class WEAT(object):

    def __init__(self, model, W=None, words_json='weat/weat.json'):
        self.model = model
        self.json = json.load(open(words_json))
        self.W = W

    def get_scores(self):
        scores = []
        for _, (name, i) in enumerate(self.json.items()):
            x_key = i['X_key']
            y_key = i['Y_key']
            a_key = i['A_key']
            b_key = i['B_key']

            score = self.__score_individual(i[x_key], i[y_key], i[a_key], i[b_key])
            scores.append(score)
        return scores

    @staticmethod
    def __balance_vectors(A, B):
        np.random.seed(2022)
        diff = len(A) - len(B)
        if diff > 0:
            A = np.delete(A, np.random.choice(range(len(A)), diff), axis=0)
        elif diff < 0:
            B = np.delete(B, np.random.choice(range(len(B)), -diff), axis=0)
        return A, B

    def __score_individual(self, X, Y, A, B):
        """
        Compute WEAT score
        :param X: target word vectors
        :param Y: target word vectors
        :param A: attribute word vectors
        :param B: attribute word vectors
        :return: WEAT score
        """

        X = self.model.transform(X)
        Y = self.model.transform(Y)
        A = self.model.transform(A)
        B = self.model.transform(B)

        X, Y = WEAT.__balance_vectors(X, Y)
        A, B = WEAT.__balance_vectors(A, B)

        # assert len(X) == len(Y) == len(A) == len(B), "{} != {} != {} != {}".format(len(X), len(Y), len(A), len(B))

        return self.__score_vectors(X, Y, A, B)

    @staticmethod
    def association_difference(W, A, B):
        return np.mean(cos_sim(W, A), axis=-1) - np.mean(cos_sim(W, B), axis=-1)

    def __score_vectors(self, X, Y, A, B):
        """
        Compute WEAT score
        :param X: target word vectors
        :param Y: attribute word vectors
        :param A: attribute word vectors
        :param B: attribute word vectors
        :return: WEAT score
        """

        x_association = WEAT.association_difference(X, A, B)
        y_association = WEAT.association_difference(Y, A, B)

        tmp1 = np.mean(x_association, axis=-1) - np.mean(y_association, axis=-1)
        tmp2 = np.std(np.concatenate((x_association, y_association), axis=0), axis=0)
        return tmp1 / tmp2