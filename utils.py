import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
from PIL import Image


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cosine_similarity(a :np.ndarray ,b :np.ndarray) ->float:
    return float(dot(a, b) / (norm(a) * norm(b)))


def load_image(image_path :str) ->Image.Image:
    return Image.open(image_path).convert("RGB")
