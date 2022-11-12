from os.path import join as j
from utils.config_utils import CONSTANTS, set_snakemake_config

DIMS = [25, 100]
WIKI = 'wiki'
PANDAS = 'pandas'
EMBEDDINGS = 'embeddings'
OUTPUT = 'output'
WORD2VEC = 'word2vec'
GLOVE = 'glove'
DATASETS = [WIKI]
DATA_SRC = {
    WIKI: '../simplewiki-20171103-pages-articles-multistream.xml.bz2',
    PANDAS: 'notebooks/rough/cleaned_ny.csv',
    EMBEDDINGS: 'embeddings',
    OUTPUT: 'data'
}

embeddings_params = {
    "threads": 55,
    "dim": 100,
    "embedding_dir": DATA_SRC[EMBEDDINGS],
    "output_dir": DATA_SRC[OUTPUT],
    "window_size": 8,
    "min_count": 10,
    "corpus_id": 0,
}

VOCAB_FILE = j("{embedding_dir}",'vocab-C0-V{min_count}.txt')
EMBEDDING_FILE = j("{embedding_dir}", 'vectors-C0-V{min_count}-W{window_size}-D{dim}-R0.05-E50-S1.bin'),
COOC_PATH = j("{embedding_dir}",'cooc-C0-V{min_count}-W{window_size}.bin')
SCORES_OUTPUT = j("{output_dir}", 'weat_scores_{dim}.npy')

rule calculate_glove_weat_scores_100_nyt:
    input:
        dataset = DATA_SRC[PANDAS],
        vocab_file = expand(VOCAB_FILE, **embeddings_params),
        embedding_file = expand(EMBEDDING_FILE, **embeddings_params),
        cooc_path = expand(COOC_PATH, **embeddings_params),
    threads: 70
    params:
        dim = 100,
        model_name = GLOVE,
        dataset_type = PANDAS
    output:
        out = expand(SCORES_OUTPUT, **embeddings_params)
    script:
        "driver_torch.py"

rule calculate_glove_weat_scores_100_dataset_wiki:
    input:
        dataset = DATA_SRC[WIKI],
        vocab_file = expand(VOCAB_FILE, **embeddings_params),
        embedding_file = expand(EMBEDDING_FILE, **embeddings_params),
        cooc_path = expand(COOC_PATH, **embeddings_params),
    threads: 55
    params:
        dataset_type = WIKI,
        model_name = GLOVE,
        dim = 100,
    output:
        out = expand(SCORES_OUTPUT, **embeddings_params)
    script:
        "driver_torch.py"

rule train_biased_word2vec_100:
    input:
        biased_iids = "dataset_100.pkl",
        dataset = DATA_SRC[WIKI]
    output:
        out = "biased_word2vec_100.bin"
    threads: 1
    run:
        # should I write code here instead of calling driver
        import pickle as pkl
        from utils.dataset import Dataset
        from models.word2vec import Word2Vec
        iids = pkl.load(open(input.biased_iids, 'rb'))
        ds = Dataset(input.dataset).lines
        model = Word2Vec(load=False, window_size=embeddings_params['window_size'],
            min_count=embeddings_params['min_count'], dim=embeddings_params['dim'])
        dataset = [ds[iid] for iid in iids]
        # TODO (ashutiwa): remove iid as mandatory parameter from base class
        model.fit(iid=None, dataset=dataset)
        model.save(output.out)

rule train_fairness_aware_word2vec:
    input:
        dataset = DATA_SRC[WIKI],
        biased_model = "biased_word2vec_100.bin"
    threads: 4
    params:
        device = "cuda:0",
        dist_metric = CONSTANTS.DIST_METRIC.DOTSIM,
        checkpoint = 1000,
        lr = .001,
        dim = 100,
        window_size = 8,
    output:
        kv_path = "kv_path.out",
        outfile = "fairness_model_ck.pt"
    run:
        set_snakemake_config(param=CONSTANTS.DIST_METRIC.__name__.lower(), value=params.dist_metric)
        set_snakemake_config(param="checkpoint", value=params.checkpoint)
        set_snakemake_config(param="lr", value=params.lr)
        set_snakemake_config(param="outfile", value=output.outfile, field_name="snakemake." + CONSTANTS.OUTPUT)

        import numpy as np
        from models.word2vec import Word2Vec
        from models.fairness_aware_model import FairnessAwareModel
        from utils.dataset import Dataset
        import gravlearn
        from tqdm import tqdm
        from utils.word2vec_sampler import Word2VecSampler

        biased_model = Word2Vec()
        biased_model.load(input.biased_model)
        num_nodes = len(biased_model._model.wv)
        dim = params.dim
        docs = Dataset(input.dataset).lines
        in_vec = np.zeros((num_nodes, dim))
        out_vec = np.zeros((num_nodes, dim))
        for i, k in enumerate(biased_model._model.wv.index_to_key):
            in_vec[i, :] = biased_model._model.wv[k]
            out_vec[i, :] = biased_model._model.syn1neg[i]
        pos_sampler = gravlearn.nGramSampler(window_length=params.window_size,
            context_window_type="double", buffer_size=1000,)
        neg_sampler = Word2VecSampler(in_vec=in_vec, out_vec=out_vec, alpha=.9, m=500)
        word2idx = biased_model._model.wv.key_to_index.copy()
        indexed_documents = [list(filter(lambda x: x != -1,map(lambda x: word2idx.get(x,-1),doc))) for doc in
                             tqdm(docs)]
        neg_sampler.fit(indexed_documents)
        pos_sampler.fit(indexed_documents)
        dataset = gravlearn.TripletDataset(epochs=1, pos_sampler=pos_sampler, neg_sampler=neg_sampler)
        dataset = gravlearn.DataLoader(dataset, batch_size=40000, shuffle=False, num_workers=4, pin_memory=True)
        model = FairnessAwareModel(device=params.device,num_nodes=num_nodes,dim=dim,params={
            "params": "snakemake.params", "output": "snakemake.output"})
        model.fit(dataset=dataset, )
        model.save(path=output.outfile, biased_wv=biased_model._model.wv, kv_path=output.kv_path)


rule test_config:
    input: dataset = DATA_SRC[WIKI],
    output: dataset = "/tmp/scores.npy"
    threads: 2
    params:
        device = "cuda:0",
        dist_metric = CONSTANTS.DIST_METRIC.DOTSIM,
        checkpoint = 1000,
        lr = .001,
    script: "test.py"

rule test_config_persistence:
    input: dataset = DATA_SRC[WIKI],
    output: dataset = "/tmp/scores.npy"
    threads: 2
    params:
        device = "cuda:0",
        dist_metric = CONSTANTS.DIST_METRIC.DOTSIM,
        checkpoint = 1000,
        lr = .001,
    run:
        import temp
        from utils.config_utils import set_snakemake_config
        set_snakemake_config(param="device", value=params.device, )
        temp.Test().func()


rule train_fairness_aware_word2vec_script:
    input:
        dataset = DATA_SRC[WIKI],
        biased_model = "biased_word2vec_100.bin"
    threads: 4
    params:
        device = "cuda:0",
        dist_metric = CONSTANTS.DIST_METRIC.DOTSIM,
        checkpoint = 1000,
        lr = .001,
        dim = 100,
        window_size = 8,
    output:
        kv_path = "kv_path.out",
        outfile = "fairness_model_ck.pt"
    script: "train_fairness_aware.py"