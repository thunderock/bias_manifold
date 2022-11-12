

from utils import glove

cooc_path = "embeddings/cooc-C0-V10-W8.bin"
embedding_dir = "embeddings"
document = "the quick brown fox jumped over the lazy dog" # add test WEAT words here and then take intersection of document and WEAT words

g = glove.Glove()
M = g.load_model(embedding_dir)
V = len(M.vocab)
X = g.load_cooc(cooc_path, V)
y = g.compute_IF_deltas(document, M, X)
print(y)

# now calculate updated word vectors



# plan:
# 1. calculate each of WEAT scores for all documents using word vectors from this method and plot index i vs WEAT score. 1 for each weat test.
# 2. parallely calculate weat score for 10 % of documents and plot index i vs WEAT score one plot for each weat test.
