from turtle import forward
from Normolization import *

import torch
import torch.nn as nn
import cv2
from torchsummary import summary
import math
import torch.optim as optim 
import torch.nn.functional as F


#Visual Encoder  
class ResNetLayer(nn.Module):

    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)


    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch

class ResNet(nn.Module):

    """
    An 18-layer ResNet architecture.
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 256, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))


    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        # print('1',batch.shape)
        batch = self.layer2(batch)
        # print('2',batch.shape)
        batch = self.layer3(batch)
        # print('3',batch.shape)
        batch = self.layer4(batch)
        # print('4',batch.shape)

        outputBatch = self.avgpool(batch)
        return outputBatch

class lipEmbeddingExtractor(nn.Module):
    def __init__(self):
        super(lipEmbeddingExtractor, self).__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(
                1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(
                1, 2, 2), padding=(0, 1, 1))
        )

        self.resnet = ResNet()


    def forward(self, inputBatch):
        # print('start',inputBatch.shape)
        # inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
        batchsize = inputBatch.shape[0]
        batch = self.frontend3D(inputBatch)
        # print(batch.shape)

        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        outputBatch = self.resnet(batch)
        outputBatch = outputBatch.reshape(batchsize, -1, 256)
        # print('shape',outputBatch.shape) #[1,length,embedding]

        outputBatch = outputBatch.transpose(0, 1).transpose(1,2)
        return outputBatch

class V_TCN(nn.Module):
    def __init__(self):
        super(V_TCN, self).__init__()
        relu = nn.ReLU()
        norm_1 = nn.BatchNorm1d(256)
        dsconv = nn.Conv1d(256, 256, 3, stride=1, padding=1,dilation=1, groups=256, bias=False)
        prelu = nn.PReLU()
        norm_2 = nn.BatchNorm1d(256)
        pw_conv = nn.Conv1d(256, 256, 1, bias=False)
        self.net = nn.Sequential(relu, norm_1 ,dsconv, prelu, norm_2, pw_conv)
    def forward(self,x):
        out = self.net(x)
        return out+x 

class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()

        self.lipembeddingextractor = lipEmbeddingExtractor()

        ve_blocks = []
        for x in range(5):
            ve_blocks +=[V_TCN()]
        self.net = nn.Sequential(*ve_blocks)

        self.liner = nn.Linear(256,5089)

    def forward(self,x):
        target_emb = self.lipembeddingextractor(x)
        # length = target_emb.shape[0]
        temp_emb = target_emb.permute(0,2,1)


        sp_embs = self.liner(temp_emb) #length,B,1500_e
        sp_embs = sp_embs.permute(1,0,2)


        out = self.net(target_emb)
        # print('out',out.shape) #length,embedding,1

        out = out.transpose(0,2)

        return out,sp_embs


#Audio Encoder
class AudioEncoder(nn.Module):
    '''
    Encoder of the TasNet
    '''
    def __init__(self):
        super(AudioEncoder,self).__init__()
        self.encoder = nn.Conv1d(1,256,40,stride=20)
        self.relu = nn.ReLU(True)
    def forward(self,x):
        out = self.encoder(x)
        out = self.relu(out)
        return out 

#Audio Decoder

class AudioDecoder(nn.Module):
    '''
    Decoder of the TasNet
    '''
    def __init__(self,):
        super(AudioDecoder,self).__init__()
        self.decoder = nn.ConvTranspose1d(256,1,40,20)
    
    def forward(self,x):
        out = self.decoder(x)
        return out 

#separation network
def select_norm(norm, dim):
    '''
    select normolization method
    norm: the one in ['gln','cln','bn']
    '''
    if norm not in ['gln', 'cln', 'bn']:
        raise RuntimeError("only accept['gln','cln','bn']")
    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    elif norm == 'cln':
        return CumulativeLayerNorm(dim, trainable=True)
    elif norm == 'bn':
        return nn.BatchNorm1d(dim)

class Conv1D_Block(nn.Module):
    '''
    sub-block with the exponential growth dilation factors 2**d
    '''

    def __init__(self, in_channels=256, out_channels=256, kernel_size=3,
                 dilation=1, norm='gln', causal=False):
        super(Conv1D_Block, self).__init__()
        # this conv1d determines the number of channels
        self.linear = nn.Conv1d(in_channels, out_channels, 1,1)  # set kernel_size=1
        self.ReLu = nn.ReLU(True)
        self.norm = select_norm(norm, out_channels)
        # keep time length unchanged
        self.pad = (dilation*(kernel_size-1))//2 if not causal else (
            dilation * (kernel_size-1))

        self.DepthwiseConv = nn.Conv1d(out_channels, out_channels,
                                    kernel_size, groups=out_channels, padding=self.pad,dilation=dilation)
        self.SeparableConv = nn.Conv1d(out_channels, in_channels, 1)
        self.causal = causal

    def forward(self, x):
        c = self.linear(x)
        
        c = self.ReLu(c)
        c = self.norm(c)
        c = self.DepthwiseConv(c)
        if self.causal:
            c = c[:, :, :-self.pad]
        c = self.SeparableConv(c)
        return x+c

class Conv1D_S(nn.Module):
    def __init__(self,num_repeats,num_blocks):
        super(Conv1D_S,self).__init__()
        self.net = self._Sequential_repeat(num_repeats=num_repeats, num_blocks=num_blocks, 
        in_channels=256, out_channels=256,
            kernel_size=3,norm='gln', causal=False)
        
    def forward(self, x):
        c = self.net(x)
        return c  #shape [-1,1,256]

    def _Sequential_repeat(self, num_repeats, num_blocks, **kwargs):
        repeat_lists = [self._Sequential_block(
            num_blocks, **kwargs) for i in range(num_repeats)]
        return nn.Sequential(*repeat_lists)

    def _Sequential_block(self, num_blocks, **kwargs):
        '''
        Sequential 1-D Conv Block
        input:
            num_blocks:times the block appears
            **block_kwargs
        '''
        Conv1D_Block_lists = [Conv1D_Block(
            **kwargs, dilation=(2**i)) for i in range(num_blocks)]
        return nn.Sequential(*Conv1D_Block_lists)

class Separation(nn.Module):
    def __init__(self,):
        super(Separation,self).__init__()
        self.Conv1D_S_pre = Conv1D_S(num_repeats=1,num_blocks=8)

        self.Conv1D_S_post = Conv1D_S(num_repeats=3,num_blocks=8)

        self.project = nn.Conv1d(256*2, 256, 1,1)  




    def forward(self,audio_em,video_em):

        audio_em = self.Conv1D_S_pre(audio_em)
        # print('audio_emb',audio_em.shape) #b,256,length
        # print('video_emb ',video_em.shape) #b,256,length 

        video_em =  F.interpolate(video_em, size=audio_em.shape[2], mode='linear')


        concat = torch.cat((audio_em,video_em),dim=1) 
        #
        projected = self.project(concat)

        out = self.Conv1D_S_post(projected)

        return out 

#AV-ConvTasNet-Spk 
class AV_ConvTasNet_Spk(nn.Module):
    
    def __init__(self,):
        super(AV_ConvTasNet_Spk,self).__init__()

        self.audioencoder = AudioEncoder()
        self.visualencoder = VideoEncoder()

        #load classification module 
        model_dict = self.visualencoder.lipembeddingextractor.state_dict()
        pretrained_model_dict = torch.load('../pretrainedmodel/Classification_face.pt')['model']
        for k,v in model_dict.items():
            if 'module.visualencoder.lipembeddingextractor.'+k in pretrained_model_dict.keys():
                model_dict[k]=pretrained_model_dict['module.visualencoder.lipembeddingextractor.'+k]
            else:
                print('not load '+str(k))
        self.visualencoder.lipembeddingextractor.load_state_dict(model_dict)
        for k, v in self.visualencoder.lipembeddingextractor.named_parameters():
            v.requires_grad=False
        model_dict = self.visualencoder.liner.state_dict()
        for k,v in model_dict.items():
            if 'module.visualencoder.liner.'+k in pretrained_model_dict.keys():
                model_dict[k]=pretrained_model_dict['module.visualencoder.liner.'+k]
            else:
                print('not load '+str(k))
        self.visualencoder.liner.load_state_dict(model_dict)
        for k, v in self.visualencoder.liner.named_parameters():
            v.requires_grad=False

        self.separation = Separation()
        self.audiodecoder = AudioDecoder()



    def forward(self,audio,video):
        
        mix_audio_emb = self.audioencoder(audio)
        video_emb,sp_emb = self.visualencoder(video)
        est_mask_emb = self.separation(mix_audio_emb,video_emb)

        est_audio_emb = F.relu(est_mask_emb)* mix_audio_emb

        est_audio = self.audiodecoder(est_audio_emb)

        return est_audio,sp_emb




def count_parameters(named_parameters):
    # Count total parameters
    total_params = 0
    part_params = {}
    for name, p in sorted(list(named_parameters)):
        n_params = p.numel()
        total_params += n_params
        part_name = name.split('.')[0]
        if part_name in part_params:
            part_params[part_name] += n_params
        else:
            part_params[part_name] = n_params

    for name, n_params in part_params.items():
        print('%s #params: %.2f' % (name, n_params/1000000))
    print("Total %.2f M parameters" % (total_params / 1000000))
    print('Estimated Total Size (MB): %0.2f' %
          (total_params * 4. / (1024 ** 2)))


if __name__ == '__main__':
    model = AV_ConvTasNet_Spk()

