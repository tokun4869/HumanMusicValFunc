import torch
from torchinfo import summary
from module.model import Model
from module.const import *

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    summary(
        Model(EXTRACTOR_REPR, HEAD_MLP, device),
        input_size = (BATCH_SIZE, 661500),
        col_names=["output_size", "num_params"]
    )

    summary(
        Model(EXTRACTOR_SPEC, HEAD_MLP, device),
        input_size = (BATCH_SIZE, 661500),
        col_names=["output_size", "num_params"]
    )

    summary(
        Model(EXTRACTOR_FEAT, HEAD_MLP, device),
        input_size = (BATCH_SIZE, 661500),
        col_names=["output_size", "num_params"]
    )