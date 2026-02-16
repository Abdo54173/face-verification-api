import torch
import torch.nn.functional as F
from typing import Tuple

def compute_distance(
    emb1: torch.Tensor,
    emb2: torch.Tensor
) -> float:
    """
    Compute cosine distance between two embeddings.
    """
    # convert to tensor if numpy
    if not isinstance(emb1, torch.Tensor):
        #emb1 = torch.tensor(emb1, dtype=torch.float32)
        emb1 = torch.tensor(emb1, dtype=torch.float32, device="cpu")

    if not isinstance(emb2, torch.Tensor):
        #emb2 = torch.tensor(emb2, dtype=torch.float32)
        emb2 = torch.tensor(emb2, dtype=torch.float32, device="cpu")

    emb1 = emb1.squeeze()
    emb2 = emb2.squeeze()

    similarity = F.cosine_similarity(emb1, emb2, dim=0).item()
    dist = 1 - similarity
    return dist


def verify_embeddings(
    emb1,
    emb2,
    threshold: float = 0.5
) -> Tuple[bool, float]:
    
    # Protection against None
    if emb1 is None or emb2 is None:
        print("One or both embeddings are None")
        return False, 1.0
    
    # Protection against NaN values
    if (
        torch.isnan(torch.tensor(emb1)).any()
        or torch.isnan(torch.tensor(emb2)).any()
    ):
        print("NaN detected in embeddings")
        return False, 1.0
    
    try:
        dist = compute_distance(emb1, emb2)
    except Exception as e:
        print(f" Error during distance computation: {e}")
        return False, 1.0

    is_match = dist <= threshold
    return is_match, dist



    
    
