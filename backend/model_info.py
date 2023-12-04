from torchinfo import summary
from module.model import ReprMPL, ReprLinearRegression, SpecCNN, FeatCNN 
from module.const import *

if __name__ == "__main__":
    summary(
        ReprMPL(),
        input_size = (BATCH_SIZE, 60),
        col_names=["output_size", "num_params"]
    )

    summary(
        ReprLinearRegression(),
        input_size = (BATCH_SIZE, 60),
        col_names=["output_size", "num_params"]
    )

    summary(
        SpecCNN(),
        input_size = (BATCH_SIZE, 1, 128, 1292),
        col_names=["output_size", "num_params"]
    )

    summary(
        FeatCNN(),
        input_size = (BATCH_SIZE, 30, 1292),
        col_names=["output_size", "num_params"]
    )