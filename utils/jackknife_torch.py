
import numpy as np
from torch.utils.data import Dataset
from utils import weat

class JackKnifeTorch(Dataset):

    def __init__(self, dataset, model) -> None:
        super().__init__()
        self.dataset = dataset
        assert self.dataset.stream is False, 'Streaming data not supported in JackKnifeTorch yet'
        self.model = model
        self.ids = list(range(self.dataset.size))

    def __getitem__(self, idx: int) -> np.array:
        W = self.model.fit(dataset=self.dataset.lines, workers=1, iid=idx)
        scorer = weat.WEAT(self.model, W)
        return np.array(scorer.get_scores()), idx
    
    def __len__(self) -> int:
        return self.dataset.size
    
