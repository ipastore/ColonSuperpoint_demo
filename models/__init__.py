"""Model factories for SuperPoint variants used in demo scripts."""

from pathlib import Path
from typing import Callable, Dict, Iterable

import torch
from torch import nn

from .MagicLeap import SuperPointNet as MagicLeapSuperPointNet

SUPERPOINT_MODEL_FACTORIES: Dict[str, Callable[[], nn.Module]] = {
    'MagicLeap': MagicLeapSuperPointNet,
    'SuperpointNet': MagicLeapSuperPointNet,
    'SuperpointNet_gauss2': MagicLeapSuperPointNet,
}


def available_superpoint_models() -> Iterable[str]:
  """Return identifiers for the supported SuperPoint model variants."""
  return SUPERPOINT_MODEL_FACTORIES.keys()


def build_superpoint_model(name: str, weights_path: str, device: torch.device) -> nn.Module:
  """Instantiate and load a SuperPoint model with pretrained weights.

  Args:
      name: Identifier for the SuperPoint architecture variant.
      weights_path: Filesystem path to the checkpoint to load.
      device: Target device where the model will run.

  Returns:
      A SuperPoint network ready for inference on the requested device.

  Raises:
      FileNotFoundError: If the weights file does not exist.
      ValueError: If the requested model name is unsupported.
  """
  if name not in SUPERPOINT_MODEL_FACTORIES:
    options = ', '.join(SUPERPOINT_MODEL_FACTORIES.keys())
    raise ValueError(f'Unknown SuperPoint model "{name}". Available options: {options}.')

  weights = Path(weights_path)
  if not weights.exists():
    raise FileNotFoundError(f'Weights file not found: {weights}')

  model = SUPERPOINT_MODEL_FACTORIES[name]()
  state_dict = torch.load(str(weights), map_location='cpu')
  model.load_state_dict(state_dict)
  model = model.to(device)
  model.eval()
  return model


SUPERPOINT_MODEL_CHOICES = tuple(SUPERPOINT_MODEL_FACTORIES.keys())

__all__ = [
    'SUPERPOINT_MODEL_CHOICES',
    'SUPERPOINT_MODEL_FACTORIES',
    'available_superpoint_models',
    'build_superpoint_model',
]
