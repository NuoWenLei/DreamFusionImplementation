import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
from stable_diffusion_tf.constants import _UNCONDITIONAL_TOKENS
from typing import Callable, Any
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import math
from PIL import Image, ImageDraw