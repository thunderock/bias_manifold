import numpy as np
from models.word2vec import Word2Vec
from models.fairness_aware_model import FairnessAwareModel
from utils.dataset import Dataset
import gravlearn
from tqdm import tqdm
from utils.word2vec_sampler import Word2VecSampler
from utils.config_utils import get_sk_value, CONSTANTS


params = snakemake.params
output = snakemake.output
input = snakemake.input
print(type(get_sk_value(CONSTANTS.DIST_METRIC.__name__.lower(), params, object=True)))
print(get_sk_value("checkpoint", params))

biased_model = Word2Vec()
biased_model.load(get_sk_value("biased_model", input))
docs = Dataset(get_sk_value("dataset", input)).lines
num_nodes = len(biased_model._model.wv)
dim = 100
in_vec = np.zeros((num_nodes, dim))
out_vec = np.zeros((num_nodes, dim))
for i, k in enumerate(biased_model._model.wv.index_to_key):
    in_vec[i, :] = biased_model._model.wv[k]
    out_vec[i, :] = biased_model._model.syn1neg[i]
pos_sampler = gravlearn.nGramSampler(window_length=8,
    context_window_type="double", buffer_size=1000,)
neg_sampler = Word2VecSampler(in_vec=in_vec, out_vec=out_vec, alpha=.9, m=500)
word2idx = biased_model._model.wv.key_to_index.copy()
indexed_documents = [list(filter(lambda x: x != -1,map(lambda x: word2idx.get(x,-1),doc))) for doc in
                     tqdm(docs)]
neg_sampler.fit(indexed_documents)
pos_sampler.fit(indexed_documents)
dataset = gravlearn.TripletDataset(epochs=1, pos_sampler=pos_sampler, neg_sampler=neg_sampler)
dataset = gravlearn.DataLoader(dataset, batch_size=40000, shuffle=False, num_workers=4, pin_memory=True)

model = FairnessAwareModel(device="cuda:0", num_nodes=num_nodes, dim=dim, params={"params": params, "output": output})
model.fit(dataset=dataset)
model_output = get_sk_value("outfile", output)
kv_path = get_sk_value("kv_path", output)
model.save(path=model_output, biased_wv=biased_model._model.wv, kv_path=kv_path)
del model