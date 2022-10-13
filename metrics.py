import torch
import numpy as np

def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = preds.eq(labels)
    acc = correct.float().mean().item()
    return acc

def P_per_image(label,topK_Class,n_q):
    '''
    topK_Class : type:numpy 
           shape: k,
    '''
    sum_rel_q = np.sum((label==topK_Class).astype(np.uint8))
    return sum_rel_q/n_q

def compute_mP_at_K(q_target_labels,database_target_labels,topK_Class,k=5):
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
    # get database total numbers
    label_flatten = np.reshape(q_target_labels,(-1))
    q_nums = np.sum((database_target_labels==label_flatten).astype(np.uint8),0) #
    q_nums = np.min((q_nums,np.ones_like(q_nums,dtype=np.uint8)*k),axis=0)
    mP_at_k = []
    for i in range(q_nums.shape[0]):
        mP_at_k.append(P_per_image(label_flatten[i],topK_Class[i,:],q_nums[i]))
    return np.mean(mP_at_k)

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
    