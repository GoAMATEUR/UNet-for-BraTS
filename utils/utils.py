import torch

def pred2onehot(prediction)->torch.Tensor:
    """get One-hot output from probability prediction

    Args:
        prediction (tensor): (N, C, H, W)
    """
    result = prediction >= 0.5
    
    # result.scatter_(dim,
    #                 prediction.argmax(dim).unsqueeze(dim),
    #                 torch.ones(2, dtype=torch.long).unsqueeze(dim))
    return result.float()

if __name__ == "__main__":
    a = torch.tensor([[[[0.1, 0.6],\
                        [0.7, 0.2]]]])
    print(pred2onehot(a))