
import numpy as np
from utils import dataset, weat
from tqdm import tqdm, trange
from models.word2vec import Word2Vec
import multiprocessing as mp


class JackKnife(object):
    def __init__(self, dataset):
        self.dataset = dataset
        assert self.dataset.stream is False, 'Streaming data not supported in JackKnife yet'

    @staticmethod
    def score_dataset_id(iid, instances, model_module):
        # print(iid)
        model = model_module()
        W = model.fit(dataset=instances, workers=1, iid=iid)
        scorer = weat.WEAT(model, W)
        return scorer.get_scores()

    def weat_scores(self, model_module=Word2Vec):
        total = self.dataset.size
        total = 3
        threads = pool_size = 2
        # score_func = partial(JackKnife.score_dataset_id, instances=self.dataset.lines)
        final_result = np.zeros((total, 7))
        st = 0
        for i in trange(total // pool_size):
            st = pool_size * i
            pool = mp.Pool(processes=threads)
            result = pool.starmap(JackKnife.score_dataset_id, [(st + j, self.dataset.lines, model_module) for j in range(pool_size)])
            pool.close()
            pool.join()
            final_result[st: st + pool_size, :] = np.array(result)
        st += pool_size
        for i in trange(st, total):
            final_result[i, :] = np.array(JackKnife.score_dataset_id(i, self.dataset.lines, model_module))
        # print(final_result, st)
        return final_result
