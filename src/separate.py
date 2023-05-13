#separate audio from the mix_audio using trained model and save the audio 
import torch
from AV-ConvTasNet import AVConvTasNet
from AV-ConvTasNet-Spk import AV_ConvTasNet_Spk
from AV-ConvTasNet-Sync import AV_ConvTasNet_Sync
from DAVSE import DAVSE

from DataLoader import make_loader
import soundfile
import numpy as np 
import os 
from Loss import cal_si_snr
import pypesq 
from tqdm import tqdm 
import torch.nn.functional as F

import torch.nn as nn
from collections import OrderedDict
import time 
import yaml 

start = time.time()

output_path = "./log_davse_lip/separate/"
model_path = "./log_davse_lip/model/Checkpoint_best.pt"

samplerate=16000

os.makedirs(output_path,exist_ok=True)


model = DAVSE()
model = nn.DataParallel(model)
checkpoint = torch.load(model_path)
model_dict = checkpoint['model']


model.load_state_dict(model_dict)
model.cuda()
model.eval()

with open('./config.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

test_loader = make_loader(params,stage='test',data_type="diff_audio_shuffle_visual",batch_size=1,\
            num_workers=1,prefetch_factor=1)

estimate_SISNR=0
estimate_PESQ=0

mix_SISNR=0
mix_PESQ=0

length=0

print('start separate >>>>')
with torch.no_grad():
    total_loss = 0 
    total_data_num=0
    for idx, data in tqdm(enumerate(test_loader)):
        audio_mix = data[0].float().cuda()
        audio_s1 = data[1].float().cuda()
        video_s1 = data[2].float().cuda()
        sp_id = data[3].cuda()


        audio_est_s1 = model(audio_mix,video_s1)
        # audio_est_s1,sp_embs = model(audio_mix,video_s1)

        # ce_loss=0
        # ce = nn.CrossEntropyLoss(reduction='mean')

        # L = sp_embs.shape[1]
        # for i in range(L):
        #     ce_loss+=ce(sp_embs[:,i,:],sp_id)
        # ce_loss/=L 

        
        if audio_est_s1.shape[2]>= audio_mix.shape[2]:
            audio_est_s1=audio_est_s1[:,:,0:audio_mix.shape[2]]
        else:
            audio_est_s1 = F.pad(audio_est_s1, pad=(0, audio_mix.shape[2]-audio_est_s1.shape[2]), value=0.0)

        # compute mixture metric loss ----------------------
        MIXSISNR=cal_si_snr(audio_s1,audio_mix).item()
        MIXPESQ = pypesq.pesq(audio_s1[0][0].cpu().detach().numpy(),audio_mix[0][0].cpu().detach().numpy(),fs=samplerate)

        mix_SISNR += MIXSISNR
        mix_PESQ+=MIXPESQ 

        ESTSISNR=0
        ESTPESQ=0

        ESTSISNR = cal_si_snr(audio_s1,audio_est_s1).item()
        audio_est_s1= audio_est_s1*torch.max(torch.abs(audio_mix))/torch.max(torch.abs(audio_est_s1))
        ESTPESQ=pypesq.pesq(audio_s1[0][0].cpu().detach().numpy(),audio_est_s1[0][0].cpu().detach().numpy(),fs=samplerate)

        estimate_SISNR += ESTSISNR
        estimate_PESQ+=ESTPESQ


        # print(idx,MIXSISNR,MIXPESQ,ESTSISNR,ESTPESQ)
        # audio_est_s1 = audio_est_s1[0].permute(1,0).cpu().detach().numpy()
        # audio_mix = audio_mix[0].permute(1,0).cpu().detach().numpy()
        # audio_s1 = audio_s1[0].permute(1,0).cpu().detach().numpy()
        # soundfile.write(output_path+'%04d_%02d_%02d_ests1.wav'%(idx,ESTSISNR,MIXSISNR),audio_est_s1,samplerate=samplerate)
        # soundfile.write(output_path+"%04d_mix.wav"%idx,audio_mix,samplerate=samplerate)
        # soundfile.write(output_path+"%04d_s1.wav"%idx,audio_s1,samplerate=samplerate)


        length+=1


    print('est_SI_SNR=%.02f, est_PESQ=%.02f'%(-estimate_SISNR/length,estimate_PESQ/length))
    print('MIX_SI_SNR=%.02f, MIX_PESQ=%.02f'%(-mix_SISNR/length,mix_PESQ/length))

    end = time.time()
    print('time= %d min'%((end-start)/60))

