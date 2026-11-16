"""Stable checkpoint classes for joint network matching and final source coordinates."""
import numpy as np
from .matching import late_vectors,harden_assignment


class SimilarityGraphModel:
    def __init__(self,source,matcher,graph_weight,iterations=5,assignment_mode='soft'):
        self.source=np.asarray(source);self.matcher=matcher;self.hosts=matcher.hosts
        self.graph_weight=graph_weight;self.iterations=iterations
        self.training_end=matcher.training_end
        self.assignment_mode=assignment_mode

    def predict(self,observation,target_hosts=None):
        assignment,unknown,info=self.matcher.assign(observation,target_hosts,
            graph_weight=self.graph_weight,iterations=self.iterations)
        if getattr(self,'assignment_mode','soft')=='hard':
            assignment,unknown=harden_assignment(assignment,unknown)
            info['assignment_mode']='hard sparse partial assignment'
            info['hard_unknown_count']=int(np.count_nonzero((observation.events>=300)&(unknown>0)))
        return late_vectors(assignment,unknown,self.source),info
