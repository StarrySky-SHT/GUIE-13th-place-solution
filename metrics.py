import torch
import numpy as np

def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = preds.eq(labels)
    acc = correct.float().mean().item()
    return acc

def P_per_image(rank):
    '''
    rank : type:numpy 
           shape: k,
    '''
    if rank.all()==0:
        return 0
    n_q = np.min((np.max(np.where(rank==1))+1,5))
    sum_rel_q = np.sum(rank)
    return sum_rel_q/n_q

def compute_mP_at_K(labels,predictions,k=5):
    '''
    labels : type:numpy 
            shape:num_query,1 
            represent the number of query
    predictions : tpye:numpy
                  shape:num_query,num_embed
                  represent the nearest labels predicted by KNN of every query image
    k   :  top k
    return: type:float
            represent the mAP@k
    '''
    predictions = predictions[:,:k]
    ranks = (predictions==labels).astype(np.uint8) #num_querry,k
    mAP_at_k = []
    for i in range(ranks.shape[0]):
        mAP_at_k.append(P_per_image(ranks[i,:]))
    return np.mean(mAP_at_k)

def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """    
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0

def map_per_set(labels, predictions):
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list
             A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    return np.mean([map_per_image(l, p) for l,p in zip(labels, predictions)])
    