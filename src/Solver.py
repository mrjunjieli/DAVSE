import time 
import torch 
from Loss import cal_si_snr,cal_snr
import os 
from DataLoader import make_loader
import pypesq 
import torch.nn.functional as F
import soundfile 
import datetime
import torch.nn as nn



class Solver(object):
    def __init__(self,args,params,model,use_gpu,optimizer,logger):
        #traindata
        self.train_loader = make_loader(params,stage='train',data_type=args.data_type,\
            batch_size=args.batch_size,num_workers=args.num_workers,prefetch_factor=5)
        #eval data
        self.dev_loader=make_loader(params,stage='dev',data_type=args.data_type,\
            batch_size=args.batch_size,num_workers=args.num_workers,prefetch_factor=5)
        

        self.args = args
        self.model = model
        self.use_gpu =use_gpu
        self.optimizer = optimizer
        self.logger = logger
        self.params = params

        self._rest()


    def _rest(self):
        self.halving = False
        if self.args.continue_from:

            checkpoint = torch.load('./'+str(self.args.log_name)+'/model/Checkpoint_last.pt')

            #load model 
            model_dict = self.model.state_dict()
            pretrained_model_dict = checkpoint['model']
            pretrained_model_dict = {k:v for k,v in pretrained_model_dict.items() if k in model_dict}
            model_dict.update(pretrained_model_dict)
            self.model.load_state_dict(model_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch=checkpoint['epoch']
            self.logger.info("*** model %d Checkpoint_last has been successfully loaded! ***"%self.start_epoch)
            self.best_val_sisnr = checkpoint['best_val_sisnr']
            self.val_no_impv = checkpoint['val_no_impv']

        else:
            self.start_epoch=0
            self.best_val_sisnr = float('inf')
            self.val_no_impv = 0
            self.logger.info("*** train from scratch ***")

    def train(self):
        self.logger.info("use SI_SNR as loss function")
        for epoch in range(self.start_epoch,self.args.num_epochs):
            self.logger.info("------------")
            self.logger.info("Epoch:%d/%d"%(epoch,self.args.num_epochs))
            #train
            #--------------------------------------
            start = time.time()
            self.model.train()
            temp_Train = self._run_one_epoch(self.train_loader,epoch,state='train')
            
            end = time.time()
            self.logger.info("Train: SI_SNR=%.02f,speaker_loss=%.02f,Time:%d minutes"%(-temp_Train['si_snr'],temp_Train['speaker_loss'],(end-start)//60))

            #validation 
            #--------------------------------------
            start = time.time()
            self.model.eval()
            with torch.no_grad():
                temp_V = self._run_one_epoch(self.dev_loader,epoch,state='dev')

            end = time.time()
            self.logger.info("Val: SI_SNR=%.02f,PESQ=%.02f,speaker_loss=%.02f,Time:%d minutes"%(-temp_V['si_snr'],temp_V['pesq'],temp_V['speaker_loss'],(end-start)//60))
            time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.logger.info(time2)



            #check whether to adjust learning rate and early stop 
            #-------------------------------------
            if temp_V['si_snr'] >= (self.best_val_sisnr-0.05):
            # if temp_V['si_snr']>=self.best_val_sisnr:
                self.val_no_impv +=1 
                if self.val_no_impv >=3:
                    self.halving =True 
                if self.val_no_impv >=6:
                    self.logger.info("No improvement for 6 epoches in val dataset, early stop")
                    break
            else:
                self.val_no_impv = 0


            # half the learning rate 
            #-----------------------------------
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr']/2
                self.optimizer.load_state_dict(optim_state)
                self.logger.info("**learning rate is adjusted from [%f] to [%f]"
                                %(optim_state['param_groups'][0]['lr']*2,optim_state['param_groups'][0]['lr']))
                self.halving = False
            
            # save the model 
            #----------------------------------
            checkpoint = {'model':self.model.state_dict(),
                                'optimizer':self.optimizer.state_dict(),
                                'epoch':epoch+1,
                                'best_val_sisnr':self.best_val_sisnr,
                                'val_no_impv':self.val_no_impv}
            torch.save(checkpoint,'./'+str(self.args.log_name)+'/model/Checkpoint_last.pt')
            self.logger.info("***save checkpoint as Checkpoint_last.pt***")

            if temp_V['si_snr'] < self.best_val_sisnr:
                self.best_val_sisnr = temp_V['si_snr']
                checkpoint = {'model':self.model.state_dict(),
                                'optimizer':self.optimizer.state_dict(),
                                'epoch':epoch+1,
                                'best_val_sisnr':self.best_val_sisnr,
                                'val_no_impv':self.val_no_impv}
                torch.save(checkpoint,"./"+str(self.args.log_name)+"/model/Checkpoint_best.pt")
                self.logger.info("***save checkpoint as Checkpoint_best.pt***")



    def _run_one_epoch(self,data_loader,epoch,state='train'):
        batch_steps = len(data_loader)
        epoch_loss={'si_snr':0,'pesq':0,'speaker_loss': 0,'acc':0}
        for step,data in enumerate(data_loader):

            audio_mix = data[0].float()
            audio_s1 = data[1].float()
            video_s1 = data[2].float()
            sp_id = data[3]

            if self.use_gpu:
                audio_mix = audio_mix.cuda()
                audio_s1 = audio_s1.cuda()
                video_s1 = video_s1.cuda()
                sp_id = sp_id.cuda()

            audio_est_s1 = self.model(audio_mix,video_s1)
            # audio_est_s1,sp_embs = self.model(audio_mix,video_s1)

            ce_loss=0
            # ce = nn.CrossEntropyLoss(reduction='mean')
            # L = sp_embs.shape[1]
            # for i in range(L):
            #     ce_loss+=ce(sp_embs[:,i,:],sp_id)
            # ce_loss/=L 


            if audio_est_s1.shape[2]>= audio_mix.shape[2]:
                audio_est_s1=audio_est_s1[:,:,0:audio_mix.shape[2]]
            else:
                audio_est_s1 = F.pad(audio_est_s1, pad=(0, audio_mix.shape[2]-audio_est_s1.shape[2]), value=0.0)

            loss_snr = cal_si_snr(audio_s1,audio_est_s1)

            loss =loss_snr
    
            epoch_loss['si_snr']+=loss_snr.item()
            epoch_loss['speaker_loss']+=ce_loss


            if state=='train':
                    
                if (step+1) %100==0:
                    print_info='Epoch: %d/%d Step: %d/%d SI-SNR: %0.2f,ce_loss:%0.2f'%(epoch,self.args.num_epochs, step,batch_steps, -epoch_loss['si_snr']/(step+1),ce_loss)

                    self.logger.info(print_info)

            if state =='train': #
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
        epoch_loss['si_snr']/=batch_steps
        epoch_loss['speaker_loss']=epoch_loss['speaker_loss']/batch_steps


        return epoch_loss

