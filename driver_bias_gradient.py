
import gc
import numpy as np
from tqdm import tqdm
from utils.dataset import Dataset, PandasDataset
from torch_geometric.data import download_url
from utils.jackknife_torch import JackKnifeTorch
from models import fast_glove, word2vec, glove
from torch.utils.data import DataLoader
from datasets import wikidataset

MODELS = {"glove": fast_glove.FastGlove}
METHODS = ["jackknife", "bias_gradient"]
model = "glove"
method = "bias_gradient"


class Args(object):
    # train and save model first
    URL = 'http://www.cs.toronto.edu/~mebrunet/simplewiki-20171103-pages-articles-multistream.xml.bz2'
    # first train the glove model
    model_ = glove.Glove()
    dataset = wikidataset.WikiDataset(URL)
    model_.fit(dataset.lines, workers=4, temp_root='embeddings/')
    model = fast_glove.FastGlove
    # model.fit(dataset.lines, temp_root='/tmp/glove/')
    outfile = 'bias_gradient.npy'
    threads = 40

jk = JackKnifeTorch(dataset=Args.dataset, model=Args.model)
total = len(jk)
print(total)
threads = Args.threads
loops = total // threads + 1
loader = DataLoader(jk, batch_size=threads, shuffle=False)
scores = np.zeros((total, 7))
print(loops)
status_loop = tqdm(loader, total=loops)

for i, scores_ in enumerate(status_loop):
    indices = scores_[1]
    for ix, idx in enumerate(indices):
        if idx < total:
            scores[idx, :] = scores_[0][ix]
    if i == loops:
        break
    if i % 100:
        np.save(Args.outfile, scores)
    del indices
    gc.collect()
    status_loop.set_description('Processing batch %d' % i)

np.save(Args.outfile, scores)




