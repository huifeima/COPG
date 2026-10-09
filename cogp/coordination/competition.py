"""Candidate-competition residual for HCCG.

Each dog-task candidate exchanges information with alternatives sharing the same
shepherd and alternatives competing for the same task. The output is a unary
score residual; feasibility remains in the original HCCG action space.
"""
from __future__ import annotations
import math
import torch
from torch import nn


def mlp(i,h,o):
    return nn.Sequential(nn.Linear(i,h),nn.SiLU(),nn.Linear(h,o))

class CandidateCompetition(nn.Module):
    def __init__(self,input_dim:int,hidden:int=64,layers:int=2):
        super().__init__()
        if min(input_dim,hidden,layers)<1: raise ValueError('invalid competition head')
        self.input_dim=input_dim;self.hidden=hidden;self.layers=layers
        self.item=mlp(input_dim,hidden,hidden)
        self.updates=nn.ModuleList([mlp(3*hidden+2,hidden,hidden) for _ in range(layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.out=nn.Linear(hidden,1)
        # Exact Base policy at initialization. Unlike a second zero gate, gradients
        # enter the output immediately and reach lower layers after the first step.
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,x:torch.Tensor,task_mask:torch.Tensor|None=None):
        """x [B,M,K,F] -> residual [B,M,K]. task_mask [B,K]."""
        if x.ndim!=4 or x.shape[-1]!=self.input_dim: raise ValueError('expected [B,M,K,F]')
        B,M,K,_=x.shape
        if task_mask is None: task_mask=torch.ones((B,K),dtype=torch.bool,device=x.device)
        if task_mask.shape!=(B,K) or task_mask.dtype!=torch.bool: raise ValueError('bad task mask')
        if bool((task_mask.sum(-1)==0).any()): raise ValueError('empty task set')
        mask=task_mask[:,None,:,None].to(x.dtype)
        h=self.item(x)*mask
        for upd,norm in zip(self.updates,self.norms):
            # Alternatives for the same dog. Exclude self candidate.
            tsum=h.sum(2,keepdim=True)
            tdeg=(task_mask.sum(-1).to(x.dtype)-1).clamp_min(1).sqrt()[:,None,None,None]
            dog_alt=(tsum-h)/tdeg
            # Competitors for the same task. Exclude same dog.
            ddeg=math.sqrt(max(1,M-1))
            task_alt=(h.sum(1,keepdim=True)-h)/ddeg
            logk=torch.log1p((task_mask.sum(-1).to(x.dtype)-1).clamp_min(0))[:,None,None,None].expand(B,M,K,1)
            logm=x.new_full((B,M,K,1),math.log1p(max(0,M-1)))
            h=norm(h+upd(torch.cat((h,dog_alt,task_alt,logk,logm),-1)))*mask
        return self.out(h).squeeze(-1)*task_mask[:,None,:].to(x.dtype)
