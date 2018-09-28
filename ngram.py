# -*- coding: utf-8 -*-
"""
@author: Fredrik Wahlberg <fredrik.wahlberg@lingfil.uu.se>
"""

import numpy as np
import random

# TODO Implement fallback models
# TODO Add cleaning function


def character_tokenizer(text):
    for t in text:
        yield t


def clean_text(text):
    pass


def ordered_ngrams(model):
    keys = model.keys()
    v = model.vectorize(keys)
    for i in np.argsort(v)[::-1]:
        yield (keys[i], v[i])

class grouping_tokenizer:
    def __init__(self, order):
        assert order > 0 and order == int(order), "order must be a strictly positive integer"
        self.order_ = order

    def tokenize(self, words):
        assert len(words) >= self.order_, "text input is too short to tokenize"
        for start in range(len(words)-self.order_+1):
            ret = [words[start]]
            for offset in range(1, self.order_):
                ret.append(words[start+offset])
            yield tuple(ret)

    def __repr__(self):
        return "grouping tokenizer of order %i" % (self.order_)
    

class NGramModel:
    def __init__(self, tokenized_text, order=1):
        """
        """
        self._ngrams = dict()
        self._order = order
        self._pred_distr = None
#        self._fallback = None
#        if order >= 2:
#            self._fallback = NGramModel([], order-1)
        self.update_model(tokenized_text)

    def update_model(self, tokenized_text):
        token_sequence = list()
        for token in tokenized_text:
            token_sequence.append(token)
            if len(token_sequence) > self._order:
                token_sequence.pop(0)
            if len(token_sequence) == self._order:
                key = tuple(token_sequence)
                if key not in self._ngrams.keys():
                    self._ngrams[key] = 1
                else:
                    self._ngrams[key] += 1
#        if self._fallback is not None:
#            self._fallback.update_model(tokenized_text)
        self._pred_distr = None

    def keys(self):
        return list(self._ngrams.keys())

    @property
    def entries_(self):
        sum_of_ngrams = 0
        for key in self._ngrams.keys():
            sum_of_ngrams += self._ngrams[key]
        return sum_of_ngrams

    def __repr__(self):
        return "%i-gram model with %i unique keys" % (self._order, len(self))

    def __len__(self):
        """Returns the number of unique keys in the model"""
        return len(self._ngrams.keys())

    def vectorize(self, codebook=None):
        """Returns the a feature vector using the space spanned by
        the codebook argument"""
        if codebook is None:
            codebook = self._ngrams.keys()
        ret = np.zeros(len(codebook))
        for i, key in enumerate(codebook):
            if key in self._ngrams.keys():
                ret[i] = self._ngrams[key]
        if self.entries_ > 0:
            ret /= self.entries_
        return ret

    def predict(self, given=None):
        if self._pred_distr is None:
            self._pred_distr = dict()
            keys = self.keys()
            for i, k in enumerate(keys):
                a = k[:-1]
                b = k[-1:]
                f = self._ngrams[k]
#                print(":", k, a, b, f)
                if a not in self._pred_distr:
                    self._pred_distr[a] = dict()
                if b not in self._pred_distr[a]:
                    self._pred_distr[a][b] = f
            for a in self._pred_distr.keys():
                # Sum occurences
                s = 0
                for b in self._pred_distr[a].keys():
                    s += self._pred_distr[a][b]
                # Normalize
                for b in self._pred_distr[a].keys():
                    self._pred_distr[a][b] /= s
        # Predict the first element in a predicted sequence
        if given is None:
            population = self.keys()
            return random.choices(population, self.vectorize(population))[0]
        else:
#            print("given", given)
            assert given in self._pred_distr
            population = list(self._pred_distr[given].keys())
            weights = [self._pred_distr[given][k] for k in population]
            # assert np.isclose(np.sum(weights), 1.0)
            return random.choices(population, weights)[0]

    def predict_sequence(self, length):
        p = self.predict()
        ret = list(p)
        while len(ret) < length:
            if self._order > 1:
                given = tuple(ret[-(self._order-1):])
            else:
                given = tuple()
            if given in self._pred_distr:
                p = self.predict(given)
            else:
                print("error")
                p = self.predict()
            ret.extend(p)
        return ret

    def union(self, other):
        """Returns the union of two ngram models as a new model."""
        ret = self.copy()
        for key in other._ngrams.keys():
            if key in ret._ngrams.keys():
                ret._ngrams[key] += other._ngrams[key]
            else:
                ret._ngrams[key] = other._ngrams[key]
        return ret

    def intersect(self, other):
        ret = self.copy()
        for key in other._ngrams.keys():
            if key in ret._ngrams.keys():
                ret._ngrams.pop(key)
            else:
                ret._ngrams[key] = other._ngrams[key]
        return ret

    def rel_comp(self, other):
        """relative complement
        A copy of self with keys from other removed"""
        ret = self.copy()
        for key in other._ngrams.keys():
            if key in ret._ngrams.keys():
                ret._ngrams.pop(key)
        return ret

    def subtract(self, other):
        """subtraction
        A copy of self with keys from other removed"""
        return self.rel_comp(other)

    def copy(self):
        """Returns a deep copy of the model"""
        from copy import deepcopy
        return deepcopy(self)


if __name__ == '__main__':
    print("Tokenizer")
    print(list(character_tokenizer("abcde")))

    print("Simple model")
    ab_model = NGramModel(character_tokenizer("aabb"), 2)
    print(ab_model)
    print(ab_model.keys())
    print(ab_model.vectorize())
    assert len(ab_model.keys()) == 3
    assert np.isclose(ab_model.vectorize()[0], 1/3)
    ba_model = NGramModel(tokenized_text=character_tokenizer("bbaa"), order=2)
    print(ba_model)
    print(ba_model.keys())
    print(ba_model.vectorize())
    union_model = ab_model.union(ba_model)
    print(union_model)
    print(union_model.keys())
    print(union_model.vectorize())
    intersect_model = ab_model.intersect(ba_model)
    print(intersect_model)
    print(intersect_model.keys())
    v = intersect_model.vectorize(codebook=union_model.keys())
    assert np.isclose(v[0], 0)
    assert np.isclose(v[1], 0.5)
    assert np.isclose(np.sum(v), 1)
    print(v)
#
    print("Larger model")
    text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Pellentesque malesuada commodo diam at vestibulum. Nam urna nisl, lobortis
dapibus arcu eu, condimentum mollis turpis. Praesent luctus efficitur nibh
ut pretium. Curabitur tincidunt bibendum neque nec tempus. Ut tellus est,
consectetur sed consequat vitae, cursus quis velit. Cras aliquam varius
lorem, eget aliquam neque venenatis id. Curabitur et odio facilisis,
commodo dolor sodales, varius lacus. Curabitur et est metus. Nunc aliquam,
odio non ultrices consequat, mi quam dictum elit, quis elementum dui ligula
et neque."""
    models = [NGramModel(character_tokenizer(text), order)
              for order in range(1, 5)]
    assert len(models[0]) == len(set(text))
    for m in models:
        print(m)
        assert len(m) == m.vectorize().shape[0]
        print("Some keys:", m.keys()[:10])
        assert np.isclose(np.sum(m.vectorize()), 1.0)

    print("ngram\t: probability")
    for ngram, prob in list(ordered_ngrams(models[3]))[:10]:
        print("%s\t: %.10f" % (ngram, prob))

    print("Predictions")
    m = models[3]
    p = m.predict()
    print(p, m.predict(p[1:]))
    print(m, "predicts:", "".join(m.predict_sequence(100)))
    m = models[0]
    print(m, "predicts:", "".join(m.predict_sequence(100)))

    # %%
    print("Equal models from generated text")
    m1 = NGramModel(character_tokenizer(text), 1)
    assert len(m1.keys()) == len(set(text))
    generated_text = "".join(m.predict_sequence(100000))
    m2 = NGramModel(character_tokenizer(generated_text), 1)
    assert len(m1.keys()) == len(m2.keys())
    k = m1.keys()
    print("RMSE:", np.sqrt(np.mean((m1.vectorize(k)-m2.vectorize(k))**2)))
    print("Mean abs.:", np.mean(np.abs(m1.vectorize(k)-m2.vectorize(k))))
