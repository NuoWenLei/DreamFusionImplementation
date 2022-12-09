import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
from stable_diffusion_tf.constants import _UNCONDITIONAL_TOKENS
from typing import Callable, Any
from matplotlib.pyplot import plt
from scipy.signal import convolve2d