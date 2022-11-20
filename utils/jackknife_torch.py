
import numpy as np
from torch.utils.data import Dataset
from utils import weat

class JackKnifeTorch(Dataset):

    def __init__(self, dataset, model) -> None:
        super().__init__()
        self.dataset = dataset

        self.model = model
        self.ids = list(range(len(self.dataset.lines)))

    def __getitem__(self, idx: int) -> np.array:
        model = self.model().fit(lines=self.dataset.lines, iid=idx)
        scorer = weat.WEAT(model)
        return np.array(scorer.get_scores()), idx
    
    def __len__(self) -> int:
        return len(self.ids)
    
