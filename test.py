from datasets import wikidataset
from models import glove

m = glove.Glove()
file = 'http://www.cs.toronto.edu/~mebrunet/simplewiki-20171103-pages-articles-multistream.xml.bz2'

lines = wikidataset.WikiDataset(file).lines

m.fit(lines)
