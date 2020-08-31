import torch
import numpy as np
from collections import defaultdict
from torch.nn import CosineSimilarity, CrossEntropyLoss
from FastAutoAugment.factory import Factory

class AVH:
    def __call__(self, model, embeddings, targets) -> list:
        """Calculate AVH scores per instance in a batch
        :param network: network object of the experiment
        :type network: network object
        :param embeddings: embeddings with size 
            [batch_size, #nodes in last but one layer of the neural network]
        :type embeddings: Any
        :param targets: Targets (Ground Truth)
        :type targets: Any
        :return: List of AVH values per instance
        """
        import ipdb; ipdb.set_trace();
        weights_list = list(model.state_dict().keys())
        weight_matrix = model.state_dict()[weights_list[-2]]
        cos = CosineSimilarity(dim=1, eps=1e-6)
        targets = targets.long()
        
        num = cos(embeddings, weight_matrix[targets])
        num = torch.clamp(num, min=-1, max=1)
        num = torch.acos(num)
       
        embeddings_ = embeddings.unsqueeze(-1).expand(-1, -1, weight_matrix.shape[0])
        den = cos(embeddings_, torch.transpose(weight_matrix, 0, 1).unsqueeze(0))
        den = torch.clamp(den, min=-1, max=1)
        den = torch.acos(den)
        den = den.sum(-1)
        
        epsilon = np.finfo(float).eps
        avh_scores = num / (den + epsilon)
        return avh_scores
    
    
class InstanceLoss:
     def __call__(self, predictions, targets) -> list:
        """Calculate loss per instance in a batch
        :param predictions: Predictions (Predicted)
        :type predictions: Any
        :param targets: Targets (Ground Truth)
        :type targets: Any
        :return: dict of losses with list of loss values per instance
        """
        criterion = CrossEntropyLoss(reduction='none')
        loss_scores = criterion(predictions, targets.long())
        return loss_scores
        
    
hardness_factory = Factory()
hardness_factory.register_builder('AVH', AVH)
hardness_factory.register_builder('instance_loss', InstanceLoss)