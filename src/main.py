#author :JunjieLi
#createtime:2021/01

from AV-ConvTasNet import AVConvTasNet
from AV-ConvTasNet-Spk import AV_ConvTasNet_Spk
from AV-ConvTasNet-Sync import AV_ConvTasNet_Sync
from DAVSE import DAVSE
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import os 
import time 
import torch.nn as nn
import logging 
import argparse
from Solver import Solver
import yaml 

#set the seed for generating random numbers. 
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)




def main(args,use_gpu,params):

    logFileName = './'+str(args.log_name)+'/train_lr'+str(args.lr)+'.log'
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(logFileName,mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info("***batch_size=%d***"%args.batch_size)



    model = DAVSE()
    count_parameters(model.named_parameters())
    model = nn.DataParallel(model)

    if use_gpu:
        model.cuda()

    optimizer = optim.Adam([{'params':model.parameters()}],lr=args.lr,weight_decay=1e-5)

    solver = Solver(args,params,model=model,use_gpu=use_gpu,optimizer=optimizer,logger=logger)
    solver.train()


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
        print('%s #params: %d' % (name, n_params/1000000))
    print("Total %.2f M parameters" % (total_params / 1000000))
    print('Estimated Total Size (MB): %0.2f' %
          (total_params * 4. / (1000000)))

if __name__=='__main__':

    parser = argparse.ArgumentParser('AVConv-TasNet')

    with open('./config.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    
    #training
    parser.add_argument('--batch_size',type=int,default=6,help='Batch size')
    parser.add_argument('--num_workers',type=int,default=10,help='number of workers to generate minibatch')
    parser.add_argument('--num_epochs',type=int,default=100,help='Number of maximum epochs')
    parser.add_argument('--lr',type=float,default=1e-3,help='Init learning rate')
    parser.add_argument('--continue_from',default=False,action='store_true')
    parser.add_argument('--data_type',type=str,default='diff_audio_sync_visual', help='three kinds of data: same_audio_sync_visual diff_audio_shuffle_visual diff_audio_sync_visual')
    parser.add_argument('--log_name',type=str,default='log', help='the folder of log')

    args = parser.parse_args()

    use_gpu= torch.cuda.is_available()
    os.makedirs('./'+str(args.log_name)+'/model/',exist_ok=True)



    main(args,use_gpu,params)
