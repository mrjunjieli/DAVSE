import torch
import torch.nn.functional as F
from mir_eval.separation import bss_eval_sources
import numpy as np 

EPS = 1e-8


def cal_si_snr(source, estimate_source):
    """Calculate SI-SNR without PIT training.
    Args:
        source: [B, C, T] a tensor
        estimate_source: [B, C, T] a tensor
        B:batch_size C: channel T:lenght of audio 
        in this case C==1  only single channel
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()

    # Step 1. Zero-mean norm
    # num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / T
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / T
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate


    # Step 2. SI-SNR without PIT
    # reshape to use broadcast
    s_target = zero_mean_target        # [B, C, T]
    s_estimate = zero_mean_estimate    # [B, C, T]
    # s_target = source
    # s_estimate = estimate_source


    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]
    # print('s_target:',pair_wise_proj[0][0][0:10])
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]

    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C]

    # si_snr = torch.mean(pair_wise_si_snr, dim=-1, keepdim = True)

    # si_snr = torch.sum(pair_wise_si_snr,dim=0)
    
    si_snr = torch.mean(pair_wise_si_snr, dim=0)

    return -si_snr



def cal_snr(source,estimate_source):
    '''
    source :[Batch,channel,length]  type: tensor

    '''
    assert source.size()==estimate_source.size()
    B,C,T = source.size()
    
    #step zero mean norm 
    mean_estimate_source = torch.sum(estimate_source,dim=2,keepdim=True)/T
    mean_souce = torch.sum(source,dim=2,keepdim=True)/T 
    source = source-mean_souce
    estimate_source = estimate_source-mean_estimate_source

    #step 2 SNR
    e_noise = estimate_source-source

    snr = 10*torch.log10(torch.sum(source**2,dim=2)/(torch.sum(e_noise**2,dim=2)+EPS)+EPS)
    return torch.sum(snr,dim=0)




def cal_sdr(source,estimate_source):
    '''
    source: [B,C,T]  source should be numpy 
    '''
    SDR = 0.
    batch_size = source.shape[0]
    for batch in range(batch_size):
        source_ = source[batch]
        estimate_source_ = estimate_source[batch]
        sdr,_,_,_ = bss_eval_sources(source_,estimate_source_)
        SDR+=sdr
    return float(SDR/batch_size)




if __name__=='__main__':
    x = torch.randn(4,1,48000)
    y = torch.randn(4,1,48000)

    # x = torch.tensor([[[2.,1.]]])
    # y = torch.tensor([[[2.,1.]]])

    a,b = cal_si_snr(y,x)

    print('snr',a,b)
    # print('snr',cal_snr_(y,x).item())
    # print('sdr',cal_sdr_(y.numpy(),x.numpy()))