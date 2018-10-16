import re
from collections import defaultdict

import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from pymystem3 import Mystem


class CVUnit:
    
    def __init__(self, _id=-1, cv_id=-1, chunk_id=-1, content='', full_content=''):
        self._id = _id
        self.cv_id = cv_id
        self.chunk_id = chunk_id
        self.content = content
        self.full_content = full_content
        self.len = len(self.content.split())

        self.w2v = None
        self.d2v = None

        self.title = '' if not self.full_content else self.full_content.split('\n')[0]
        self.desc = '' if not self.full_content else self.full_content.split('\n\n')[5].strip()

    def __str__(self):
        return 'CVUnit(id=%s, cv_id=%s, chunk_id=%s, title=%s)' \
                % (self._id, self.cv_id, self.chunk_id, self.title)

    __repr__ = __str__

    @staticmethod
    def from_content(content):
        return CVUnit(-1, -1, -1, content=content)

    def set_w2v(self, model):
        vectors = list()

        for word in self.content.split():
            if word in model.wv:
                vectors.append(model.wv[word])

        if vectors:
            self.w2v = np.mean(np.array(vectors), axis=0)
        else:
            self.w2v = np.random.normal(scale=0.01, size=[model.vector_size,])

        del vectors

    def set_d2v(self, model):
        self.d2v = model.infer_vector(self.content.split())


class CVCorpus:
    """
    normalize: bool: apply text normalization
    to_lower: bool: apply text lowering
    remove_stopwords: bool: remove stopwords
    split_by: [None, 'sentences', 'words']: chunkize text by sentences, words if not None
    n_chunks: int: number of units per each chunk:
                    e.g. setting `split_by` = `sentences` and `n_chunks` = `5`
                         would chunkize each document into fragments of 5 sentences
    w2v_model: Model: model to infer w2v vectors
    d2v_model: Model: model to infer d2v vectors
    """

    def __init__(self, normalize=True,
                 to_lower=True,
                 remove_stopwords=True,
                 split_by=None,
                 n_chunks=1,
                 w2v_model=None,
                 d2v_model=None):
        self.normalize = normalize
        self.to_lower = to_lower
        self.remove_stopwords = remove_stopwords
        self.split_by = split_by
        self.n_chunks = n_chunks
        self.w2v_model = w2v_model
        self.d2v_model = d2v_model

        self.mystem = Mystem(entire_input=False)
        self.STOPWORDS = stopwords.words('russian') if self.remove_stopwords else list()

        self.SPLITTERS_MAP = {'sentences': self.split_sentences,
                              'words': self.split_words
                             }

        self.ALGOS_SIM_MAP = {'bm25': self.sim_bm25,
                              'w2v': self.sim_cosine,
                              'd2v': self.sim_cosine,
                              'blend': self.sim_blend
                             }

        self.ALGOS_CHECK_MAP = {'bm25': self.check_bm25,
                                'w2v': self.check_w2v,
                                'd2v': self.check_d2v,
                                'blend': self.check_blend
                               }

    def normalizer(self, text):
        text = text.lower() if self.to_lower else text

        if not self.normalize:
            return text

        text_norm = list()

        for word in text.split():
            word_norm = re.sub('[\W]|[\d]|_|u', '', word)

            if word_norm and word_norm not in self.STOPWORDS:
                text_norm.append(word_norm)

        return ' '.join(text_norm)

    def split_sentences(self, text):
        sents = sent_tokenize(text)

        i = 0
        j = self.n_chunks

        while i < len(sents):
            chunk = ' '.join(sents[i:j])
            chunk = self.normalizer(chunk)

            if chunk:
                yield chunk

            i += self.n_chunks
            j += self.n_chunks


    def split_words(self, text):
        i = 0
        j = self.n_chunks

        text = self.normalizer(text)
        _split = text.split()

        while i < len(_split):
            chunk = _split[i:j]
            chunk.extend(['NULL' for _ in range(self.n_chunks - len(chunk))])
            chunk = ' '.join(chunk)

            if chunk:
                yield chunk

            i += self.n_chunks
            j += self.n_chunks

    def build(self, contents, full_cvs):
        self.D = list()
        self.CVS = list()
        self._ID = 0

        for cv_id, cv_content in enumerate(contents):
            self.CVS.append(cv_content)

            if self.split_by is not None:
                splitter = self.SPLITTERS_MAP[self.split_by]

                for chunk_id, chunk_text in enumerate(splitter(cv_content)):
                    cv_unit = CVUnit(_id=self._ID,
                                     cv_id=cv_id,
                                     chunk_id='%s_%s' % (cv_id, chunk_id),
                                     content=chunk_text,
                                     full_content=full_cvs[cv_id]
                                    )
                    self.set_vectors(cv_unit)

                    self.D.append(cv_unit)
                    self._ID += 1

            else:
                content = self.normalizer(cv_content)
                cv_unit = CVUnit(_id=self._ID,
                                 cv_id=cv_id,
                                 chunk_id=-1,
                                 content=content,
                                 full_content=full_cvs[cv_id]
                                )
                self.set_vectors(cv_unit)

                self.D.append(cv_unit)
                self._ID += 1

    def build_inverted_index(self):
        self.INV_IDX = defaultdict(dict)

        for i, unit in enumerate(self.D):
            for word in unit.content.split():
                if self.INV_IDX[word].get(i) is None:
                    self.INV_IDX[word][i] = 1

                else:
                    self.INV_IDX[word][i] += 1

    def set_vectors(self, unit):
        if self.w2v_model is not None:
            unit.set_w2v(self.w2v_model)

        if self.d2v_model is not None:
            unit.set_d2v(self.d2v_model)

    def check_bm25(self):
        if 'INV_IDX' not in self.__dict__:
            self.build_inverted_index()

        if 'bm25_ready' not in self.__dict__:
            self.prepare_bm25()

    def check_w2v(self):
        if self.w2v_model is None:
            raise Exception('w2v model not found')

    def check_d2v(self):
        if self.d2v_model is None:
            raise Exception('d2v model not found')

    def check_blend(self):
        self.check_bm25()
        self.check_w2v()

    def prepare_bm25(self):
        self.k1 = 2.0
        self.b = 0.75
        self.N = len(self.D)
        self.avgdl = np.mean([unit.len for unit in self.D])
        self.bm25_ready = True

    def compute_bm25(self, qf, dl, n):
        first = (self.N - n + 0.5) / (n + 0.5)
        second = (self.k1 + 1) * qf
        third = qf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
        return np.log(first) * (second / third)

    def sim_bm25(self, query_unit, unit, model):
        score = 0
        dl = unit.len

        for word in query_unit.content.split():
            if word in self.INV_IDX:
                qf = self.INV_IDX[word][unit._id] if self.INV_IDX[word].get(unit._id) is not None else 0
                n = len(self.INV_IDX[word])
                score += self.compute_bm25(qf, dl, n)

        return score

    def sim_cosine(self, query_unit, unit, model):
        _dot = np.dot(query_unit.__dict__[model], unit.__dict__[model])
        lnorm = np.sqrt(np.sum(query_unit.__dict__[model]**2))
        rnorm = np.sqrt(np.sum(unit.__dict__[model]**2))
        return _dot / (lnorm * rnorm)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sim_blend(self, query_unit, unit, model):
        return (0.5 * self.sigmoid(self.sim_bm25(query_unit, unit, model)) + \
                0.5 * self.sim_cosine(query_unit, unit, 'w2v')) / 2

    def run_search(self, query_unit, algo):
        self.ALGOS_CHECK_MAP[algo]()

        similarity = self.ALGOS_SIM_MAP[algo]

        scores = list()

        for i, unit in enumerate(self.D):
            scores.append((unit, similarity(query_unit, unit, algo)))

        return scores

    def sort_output(self, scores, top_n):
        return [(unit, score) for unit, score in sorted(scores, key=lambda x: x[1], reverse=True)[:top_n] if score > 0]

    def prepare_query_unit(self, query):
        query = self.normalizer(query)
        query_unit = CVUnit.from_content(query)
        self.set_vectors(query_unit)
        return query_unit

    def search(self, query, top_n=5, algo='w2v', lemmatize=False, split_by=None):
        """
        query: str: query
        top_n: int: return `top_n` best results
        algo: str: ['bm25', 'w2v', 'd2v', 'blend']: algorithm used (blend=bm25+w2v)
        split_by: str: [None, 'sentences', 'words']: `query` splitting method
        """

        query = ' '.join(self.mystem.lemmatize(query)) if lemmatize else query

        if split_by is None:
            query_unit = self.prepare_query_unit(query)
            scores = self.sort_output(self.run_search(query_unit, algo), top_n)

        else:
            query_units = list()
            scores = dict()

            splitter = self.SPLITTERS_MAP[split_by]

            for chunk_id, chunk_text in enumerate(splitter(query)):
                cv_unit = CVUnit(_id=None,
                                 cv_id=None,
                                 chunk_id=None,
                                 content=chunk_text
                                 )
                self.set_vectors(cv_unit)

                query_units.append(cv_unit)

            for query_unit in query_units:
                scores.update(
                    {unit._id: score for unit, score in  self.sort_output(self.run_search(query_unit, algo), top_n)}
                             )

            scores = [(self.D[k], scores[k]) for k in sorted(scores, key=scores.get, reverse=True)[:top_n]]

        return scores
