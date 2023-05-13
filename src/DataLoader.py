import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchaudio as ta
import random
import librosa 
import sys
import os 
import soundfile as sf 
from mir_eval.separation import bss_eval_sources
import yaml
import cv2 
from Loss import cal_si_snr,cal_snr


    
def read_file(file_path):
    file_list=[]
    with open(file_path,'r') as p:
        for line in p.readlines():
            file_list.append(line.strip())
    return file_list

def shuffle_along_depth(a):
    new_array = np.zeros(a.shape)
    depth = list(range(a.shape[1]))
    random.shuffle(depth)
    for i in range(len(depth)):
        new_array[:,i,:,:] = a[:,depth[i],:,:]
    return new_array

class MyDataset(Dataset):
    def __init__(self, yaml_params, stage='train', data_type='same'):
        super(MyDataset, self).__init__()

        # stage:train,dev,test
        self.params = yaml_params
        self.fps = yaml_params['data']['fps']
        self.sr = yaml_params['data']['sr']
        self.stage = stage


        if data_type=='same_audio_sync_visual':
            if stage=='train':
                self.data_list=read_file(yaml_params['data']['same_audio_sync_visual']['train'])
            elif stage=='dev':
                self.data_list=read_file(yaml_params['data']['same_audio_sync_visual']['dev'])
            elif stage=='test':
                self.data_list=read_file(yaml_params['data']['same_audio_sync_visual']['test'])
            else:
                raise KeyError("stage must in ['train','dev','test']")
            self.V_shuffle=yaml_params['data']['same_audio_sync_visual']["V_shuffle"]
        elif data_type=='diff_audio_shuffle_visual':
            if stage=='train':
                self.data_list=read_file(yaml_params['data']['diff_audio_shuffle_visual']['train'])
            elif stage=='dev':
                self.data_list=read_file(yaml_params['data']['diff_audio_shuffle_visual']['dev'])
            elif stage=='test':
                self.data_list=read_file(yaml_params['data']['diff_audio_shuffle_visual']['test'])
            else:
                raise KeyError("stage must in ['train','dev','test']")
            self.V_shuffle=yaml_params['data']['diff_audio_shuffle_visual']["V_shuffle"]
            
        elif data_type=="diff_audio_sync_visual":
            if stage=='train':
                self.data_list=read_file(yaml_params['data']['diff_audio_sync_visual']['train'])
            elif stage=='dev':
                self.data_list=read_file(yaml_params['data']['diff_audio_sync_visual']['dev'])
            elif stage=='test':
                self.data_list=read_file(yaml_params['data']['diff_audio_sync_visual']['test'])
            else:
                raise KeyError("stage must in ['train','dev','test']")
            self.V_shuffle=yaml_params['data']['diff_audio_sync_visual']["V_shuffle"]
        else:
            raise KeyError("ERROR: there are only three kinds of \
            data: diff_audio_sync_visual diff_audio_shuffle_visual same_audio_sync_visual") 
        
        self.sp_list = self.load_sp('/Work21/2020/lijunjie/AV_ConvTasNet/data/speaker_list')


    def __len__(self):
        return len(self.data_list)

    def load_sp(self,path):
        temp_list=[]
        with open(path,'r') as p:
            for line in p.readlines():
                temp_list.append(line.strip())
        return temp_list


    def load_data_from_datalist(self,data_list,index,sr):
        mix_wav_path, s1_wav_path, s2_wav_path,\
            face1_path, face2_path, lip1_path, lip2_path = data_list[index].split(' ')

        #we only set s1 as our target speaker
        mix_wav,_ = librosa.load(mix_wav_path,sr)
        mix_wav = np.expand_dims(mix_wav, axis=0)

        s1_wav,_ = librosa.load(s1_wav_path,sr)
        s1_wav = np.expand_dims(s1_wav, axis=0)


        if s1_wav.shape[1]!= mix_wav.shape[1]:
            raise RuntimeError('the lenght of mix and s1 should be the same')
        
        lip1_data  = np.load(face1_path)
        lip1_data = np.expand_dims(lip1_data, axis=0)  #1,d,w,h
        lip1_data = lip1_data[:,0:int(self.fps/self.sr*mix_wav.shape[1]),:,:]


        if self.stage=='train' or self.stage=='dev':
            speaker = face1_path.split('/')[-2]
            sp_index = self.sp_list.index(speaker)
        else:
            sp_index=0



        # print(lip1_data.shape) #1,depth,112,112
        # image = lip1_data[:,10,:,:]
        # cv2.imwrite('abc.jpg',image[0])
        
        if self.V_shuffle:
            
            if self.stage=='train' or self.stage=='test':
                lip1_data = lip1_data.transpose(1,0,2,3)
                np.random.shuffle(lip1_data)
                lip1_data = lip1_data.transpose(1,0,2,3)


        # image = lip1_data[:,10,:,:]
        # cv2.imwrite('ab.jpg',image[0])

        return mix_wav, s1_wav, lip1_data,sp_index
    

    def __getitem__(self, index):

        mix_wav, s1_wav, visual1_data,sp_index = self.load_data_from_datalist(self.data_list,index,self.sr)

        return mix_wav, s1_wav, visual1_data,sp_index

    

def collate_fn_(dataBatch):
    mix_wav = [torch.from_numpy(batch[0]) for batch in dataBatch]
    s1_wav = [torch.from_numpy(batch[1]) for batch in dataBatch]
    visual1_data = [torch.from_numpy(batch[2])  for batch in dataBatch]
    # visual2_data = [torch.from_numpy(batch[3])  for batch in dataBatch]

    spid = [batch[-1] for batch in dataBatch]
    spid = torch.LongTensor(spid)

    Alength_list = []
    for item in mix_wav:
        Alength_list.append(item.shape[1])
    Amax_length = np.max(Alength_list)

    Vlength_list = []
    for item in visual1_data:
        Vlength_list.append(item.shape[1])
    # Vlength_list_2 = []
    # for item in visual2_data:
    #     Vlength_list_2.append(item.shape[1])


    Vmax_length = np.max(Vlength_list)

    for idx, data in enumerate(mix_wav):

        padded_data = F.pad(data, pad=(
             0, Amax_length-Alength_list[idx],0,0), value=0.0).unsqueeze(0)
        mix_wav[idx] = padded_data
    for idx, data in enumerate(s1_wav):
        padded_data = F.pad(data, pad=(
             0, Amax_length-Alength_list[idx],0,0), value=0.0).unsqueeze(0)
        s1_wav[idx] = padded_data
    
    for idx, data in enumerate(visual1_data):
        padded_data = F.pad(data, pad=(
            0, 0, 0,0,0, Vmax_length-Vlength_list[idx]), value=0.0).unsqueeze(0)
        visual1_data[idx] = padded_data


    
    visual1_data = torch.cat(visual1_data, dim=0)
    mix_wav = torch.cat(mix_wav, dim=0)
    s1_wav = torch.cat(s1_wav, dim=0)


    return mix_wav,s1_wav,visual1_data,spid


class MyLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(MyLoader, self).__init__(*args, **kwargs)


def make_loader(yaml_params, num_workers=1, batch_size=1, prefetch_factor=10, 
            stage='dev',data_type='diff_audio_shuffle_visual'):
    dataset = MyDataset(yaml_params, stage=stage,data_type=data_type)

    if stage == 'train':
        shuffle = True
    else:
        shuffle = False
    dataloader = MyLoader(dataset, num_workers=num_workers, batch_size=batch_size,\
                          prefetch_factor=prefetch_factor, shuffle=shuffle, drop_last=True, \
                          pin_memory=True,collate_fn=collate_fn_,)

    return dataloader

if __name__=='__main__':
    from tqdm import tqdm 
    with open('./config.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    data_loader = make_loader(params,stage='train',prefetch_factor=1,batch_size=2)
    snr=0
    for idx,data in tqdm(enumerate(data_loader)):
        audio_mix=data[0]
        audio_s1 = data[1]
        sp_id = data[3]
        visual1_data=data[2]

        print(audio_mix.shape) # (B,1,L)
        print(audio_s1.shape) # (B,1,L)
        print(sp_id) # list[]
        print(visual1_data.shape) #B,1,Depth,112,112
        
        break

