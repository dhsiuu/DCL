import os
import random

import numpy as np
import torch

from utils import Configurator
from DCL import DCL


def _set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    config = Configurator('./', './dataset/')

    config.add_config("./conf/training_config.ini", section="training_config")
    config.add_config("./conf/model_config.ini", section="model_config")
    config.parse_cmd()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config["gpu_id"])
    _set_random_seed(config["seed"])

    recommender = DCL(config)
    recommender.train_model()
