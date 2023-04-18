import sys
sys.path.append('..')

import torch
from torch import autograd, optim, nn

import fewshot_re_kit


class Proto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, dot=False):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.dot = dot
        self.hidden_size = 768
        self.alpha1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
    
    def __euclid_dist__(self, x, y, dim):
        return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_euclid_dist__(self, S, Q):
        return self.__euclid_dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, rel_txt, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        rel_gol, rel_loc = self.sentence_encoder(rel_txt, cat=False)
        rel_loc = torch.mean(rel_loc, 1) #[B*N, D]
        rel_rep = torch.cat((rel_gol, rel_loc), -1)
        
        support_h, support_t,  s_loc = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query_h, query_t,  q_loc = self.sentence_encoder(query) # (B * total_Q, D)
        support = torch.cat((support_h, support_t), -1)
        query = torch.cat((query_h, query_t), -1)
        support = support.view(-1, N, K, self.hidden_size*2) # (B, N, K, D)
        query = query.view(-1, total_Q, self.hidden_size*2) # (B, total_Q, D)
        
        # Prototypical Networks 
        # Ignore NA policy
        #support = torch.mean(support, 2) # Calculate prototype for each class
        #query-guided prototype enhancement
        support_ = support.view(-1, N * K, self.hidden_size * 2)
        dist_sq = self.__batch_euclid_dist__(support_, query).view(-1, total_Q, N, K)
        query_guided_weights = dist_sq.mean(1).tanh().softmax(-1).unsqueeze(-1)
        support = (support * query_guided_weights).sum(2)
        
        rel_rep = rel_rep.view(-1, N, rel_gol.shape[1]*2)
        support = self.alpha1 * support + self.alpha2 * rel_rep
        rel_gate = torch.sigmoid(self.gate2(torch.cat((rel_rep, support), -1)))
        support = torch.sigmoid(self.gate1(torch.cat((rel_gate * rel_rep, support), -1))) * support + rel_rep * (1 - rel_gate)
        
        logits = self.__batch_dist__(support, query) # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        
        return logits, pred
