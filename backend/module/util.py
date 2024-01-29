import random
import numpy as np
import torch
from module.model import Model
from module.const import SEED

def torch_fix_seed(seed=SEED):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


class EarlyStop:
    def __init__(
        self,
        model_path: str,
        threshold: int=10,
    ) -> None:

        self.model_path = model_path
        self.threshold = threshold
        self.best_loss = None
        self.count = 0
        self.stop_flag = False
    

    def __call__(
        self,
        loss: float,
        model: Model
    ) -> bool:

        if self.best_loss is None:
            self.best_update(loss=loss, model=model)
        elif self.best_loss > loss:
            self.best_update(loss=loss, model=model)
        else:
            self.count_up()
        
        return self.stop_flag
    

    def best_update(
        self,
        loss: float,
        model: Model
    ) -> None:

        self.best_loss = loss
        self.count = 0
        self.checkpoint(model=model)
    

    def checkpoint(
        self,
        model: Model
    ) -> None:

        torch.save(model.state_dict(), self.model_path)


    def count_up(
        self,
    ) -> None:
        self.count += 1
        self.stop_flag = self.count >= self.threshold