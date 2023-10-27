

"""Convenience functions for reading data."""

import io
import os
from typing import List
from net.model import utils
import haiku as hk
import numpy as np
# Internal import (7716).


def casp_model_names(data_dir: str) -> List[str]:
  params = os.listdir(os.path.join(data_dir, 'params'))
  return [os.path.splitext(filename)[0] for filename in params]


def get_model_haiku_params(model_name: str, data_dir: str) -> hk.Params:
  """Get the Haiku parameters from a model name."""

  path = os.path.join(data_dir, 'params', f'params_{model_name}.npz')

  with open(path, 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)

  return utils.flat_params_to_haiku(params)
