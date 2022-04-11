import imp
from multiprocessing import reduction
import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class LearnableSigmoid(nn.Module):
    def __init__(self, ):
        super(LearnableSigmoid, self).__init__()
        self.alpha_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.alpha_weight.data.fill_(1.0)
        self.beta_weight.data.fill_(0)
        
    def forward(self, input):
        return 1/(1 +  torch.exp(-self.alpha_weight*(input + self.beta_weight)))

class DistilE_Loss(nn.Module):
    """Negative sampling loss with self-adversarial training. 

    Attributes:
        args: Some pre-set parameters, such as self-adversarial temperature, etc. 
        model: The KG model for training.
    """
    def __init__(self, args, model, model_tea = None):
        super(DistilE_Loss, self).__init__()
        self.args = args
        self.model = model
        self.model_tea = model_tea
        self.distance = nn.SmoothL1Loss(reduction='none')
        self.sigmoid_pos_stu = LearnableSigmoid()
        self.sigmoid_neg_stu = LearnableSigmoid()
        self.sigmoid_pos_tea = LearnableSigmoid()
        self.sigmoid_neg_tea = LearnableSigmoid()

    def forward(self, pos_score, neg_score, pos_score_tea = None, neg_score_tea = None, 
                head_emb_stu = None, tail_emb_stu = None, head_emb_tea = None, tail_emb_tea = None, 
                subsampling_weight = None): 
        """Negative sampling loss with self-adversarial training. In math:
        
        L=-\log \sigma\left(\gamma-d_{r}(\mathbf{h}, \mathbf{t})\right)-\sum_{i=1}^{n} p\left(h_{i}^{\prime}, r, t_{i}^{\prime}\right) \log \sigma\left(d_{r}\left(\mathbf{h}_{i}^{\prime}, \mathbf{t}_{i}^{\prime}\right)-\gamma\right)
        
        Args:
            pos_score: The score of positive samples.
            neg_score: The score of negative samples.
            subsampling_weight: The weight for correcting pos_score and neg_score.

        Returns:
            loss: The training loss for back propagation.
        """

        
        # loss of distilE 
        # eq4 𝑑𝑆𝑐𝑜𝑟𝑒 = 𝑙𝛿 ( 𝑓 𝑇𝑟 (ℎ, 𝑡 ), 𝑓 𝑆𝑟 (ℎ, 𝑡 )).
        d_score_pos = self.distance(torch.sigmoid(pos_score), torch.sigmoid(pos_score_tea)) # 512, 1
        d_score_neg = self.distance(torch.sigmoid(neg_score), torch.sigmoid(neg_score_tea)) # 512, 1024

        # eq 5
        head_norm_stu = torch.norm(head_emb_stu, p=2,dim=1, keepdim=True) # (512,1)
        tail_norm_stu = torch.norm(tail_emb_stu, p=2,dim=1, keepdim=True) # (512,1)

        head_norm_tea = torch.norm(head_emb_tea, p=2,dim=1, keepdim=True) # (512,1)
        tail_norm_tea = torch.norm(tail_emb_tea, p=2,dim=1, keepdim=True) # (512,1)

        angle_stu = ((head_emb_stu/head_norm_stu) * (tail_emb_stu/tail_norm_stu)).sum(dim = -1,keepdim=True) # [512,1])
        angle_tea = ((head_emb_tea/head_norm_tea) * (tail_emb_tea/tail_norm_tea)).sum(dim = -1,keepdim=True) # [512,1])

        LenRat_stu = head_norm_stu/tail_norm_stu # 512, 1 
        LenRat_tea = head_norm_tea/tail_norm_tea # 512, 1

        d_structure = self.distance(angle_stu, angle_tea) + self.distance(LenRat_stu, LenRat_tea) # 512, 1

        # eq7 
        d_soft_pos = d_score_pos # + d_structure # 512, 1
        d_soft_neg = d_score_neg # 512, 1024

        # eq8 & eq9
        p_possoft_stu = self.sigmoid_pos_stu(pos_score_tea).squeeze(1) # 512
        p_negsoft_stu = 1 - self.sigmoid_pos_stu(neg_score_tea) # 512,1024

        # eq14 & eq15
        if self.args.stage2:
            p_possoft_tea = self.sigmoid_pos_tea(pos_score).squeeze(1) # 512
            p_negsoft_tea = 1 - self.sigmoid_neg_tea(neg_score) # 512, 1024
        # ---------------------------------------------------------------
        # 把1024负样本合成一个样本 
        if self.args.negative_adversarial_sampling:
            weight_neg_sampl = F.softmax(neg_score * self.args.adv_temp, dim=1).detach()
            neg_score_hl = (weight_neg_sampl * F.logsigmoid(-neg_score)).sum(dim=1)  #shape:[bs]
            d_soft_neg = (weight_neg_sampl * d_soft_neg).sum(dim=1) #负样本软标签
            p_negsoft_stu = (weight_neg_sampl * p_negsoft_stu).sum(dim=1) #stu负样本软标签权重
            if self.args.stage2:
                neg_score_tea_hl = (F.softmax(neg_score_tea * self.args.adv_temp, dim=1).detach() * F.logsigmoid(-neg_score_tea)).sum(dim=1)  #shape:[bs]
                p_negsoft_tea = (weight_neg_sampl * p_negsoft_tea).sum(dim=1)
            
        else:
            neg_score_hl = F.logsigmoid(-neg_score).mean(dim = 1) # 计算hard loss的neg_score
            d_soft_neg = d_soft_neg.mean(dim = 1)
            p_negsoft_stu = p_negsoft_stu.mean(dim = 1)
            if self.args.stage2:
                neg_score_tea_hl = F.logsigmoid(-neg_score_tea).mean(dim = 1)
                p_negsoft_tea = p_negsoft_tea.mean(dim = 1)

        # pos_score_ori = pos_score.clone() #clone是深拷贝，共享梯度 detach()函数可以返回一个完全相同的tensor,与旧的tensor共享内存, 脱离计算图 
        pos_score_hl = F.logsigmoid(pos_score).view(pos_score.shape[0]) #shape:[bs] # hard 
        d_soft_pos = d_soft_pos.squeeze(1)
        if self.args.stage2:
            pos_score_tea_hl = F.logsigmoid(pos_score_tea).view(pos_score_tea.shape[0]) #shape:[bs]

        
        # 原始loss
        # if self.args.use_weight:
        #     positive_sample_loss = - (subsampling_weight * pos_score).sum()/subsampling_weight.sum()
        #     negative_sample_loss = - (subsampling_weight * neg_score).sum()/subsampling_weight.sum()
        # else:
        #     positive_sample_loss = - pos_score.mean()
        #     negative_sample_loss = - neg_score.mean()

        # loss = (positive_sample_loss + negative_sample_loss) / 2

        # import pdb; pdb.set_trace()
        # eq10
        if self.args.use_weight:
            Lsoft_stu = (subsampling_weight*p_possoft_stu*(d_soft_pos)).sum()/subsampling_weight.sum() + (subsampling_weight * p_negsoft_stu*(d_soft_neg)).sum()/subsampling_weight.sum()
            Lhard_stu = (subsampling_weight*(1-p_possoft_stu)*(-pos_score_hl)).sum()/subsampling_weight.sum() + (subsampling_weight * (1-p_negsoft_stu)*(-neg_score_hl)).sum()/subsampling_weight.sum()
            if self.args.stage2:
                Lsoft_tea = (subsampling_weight*p_possoft_tea*(d_soft_pos)).sum()/subsampling_weight.sum() + (subsampling_weight * p_negsoft_tea*(d_soft_neg)).sum()/subsampling_weight.sum()
                Lhard_tea = (subsampling_weight*(1-p_possoft_tea)*(-pos_score_tea_hl)).sum()/subsampling_weight.sum() + (subsampling_weight * (1-p_negsoft_tea)*(-neg_score_tea_hl)).sum()/subsampling_weight.sum()
                
        else:
            Lsoft_stu = (p_possoft_stu*(d_soft_pos)).mean() + (p_negsoft_stu*(d_soft_neg)).mean()
            Lhard_stu = ((1-p_possoft_stu)*(-pos_score_hl)).mean() + ((1-p_negsoft_stu)*(-neg_score_hl)).mean()
            if self.args.stage2:
                Lsoft_tea = (p_possoft_tea*(d_soft_pos)).mean() + (p_negsoft_tea*(d_soft_neg)).mean()
                Lhard_tea = ((1-p_possoft_tea)*(-pos_score_tea_hl)).mean() + ((1-p_negsoft_tea)*(-neg_score_tea_hl)).mean()
        if self.args.stage2:
            loss = Lsoft_stu + Lhard_stu + Lsoft_tea + Lhard_tea
            #print(f'loss{loss}, sl{Lsoft_stu}, hl{Lhard_stu}, tsl{Lsoft_tea}, thl{Lhard_tea}')
        else:
            loss = Lsoft_stu + Lhard_stu
            #print(f'loss{loss}, sl{Lsoft_stu}, hl{Lhard_stu}')
        #loss = Lhard_stu

        

        if self.args.model_name == 'ComplEx' or self.args.model_name == 'DistMult' or self.args.model_name == 'BoxE':
            #Use L3 regularization for ComplEx and DistMult
            regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p = 3)**3 + \
                self.model.rel_emb.weight.norm(p = 3)**3
            )
            # embed();exit()
            if self.args.stage2:
                regularization += self.args.regularization * (
                self.model_tea.ent_emb.weight.norm(p = 3)**3 + \
                self.model_tea.rel_emb.weight.norm(p = 3)**3
            )
            loss = loss + regularization
        return loss
    
    def normalize(self):
        """calculating the regularization.
        """
        regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p = 3)**3 + \
                self.model.rel_emb.weight.norm(p = 3)**3
            )
        if self.args.stage2:
                regularization += self.args.regularization * (
                self.model_tea.ent_emb.weight.norm(p = 3)**3 + \
                self.model_tea.rel_emb.weight.norm(p = 3)**3
            )
        return regularization