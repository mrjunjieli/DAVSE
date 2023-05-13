#!/bin/bash 
#

#this file is used to generate data file path for pretrian trainval and test 

audio_data_folder_d='/CDShare3/LRS3_process/Simulated/mixaudio_16k_d/min'  #mixed audio from different spks 
audio_data_folder_s='/CDShare3/LRS3_process/Simulated/mixaudio_16k_s/min' #mixed audio from same spk 
faces_data_folder='/CDShare3/LRS3_process/faces_npy'
lips_data_folder='/CDShare3/LRS3_process/lips_npy'
cutted_audio_folder='/CDShare3/LRS3_process/Simulated/cutted'  


for kind in `ls $audio_data_folder_d`
do 
    mix_path=$audio_data_folder_d/$kind/mix
    s1_path=$audio_data_folder_d/$kind/s1 
    s2_path=$audio_data_folder_d/$kind/s2 

    for wav in `ls $mix_path`
    do 

        speaker1=`echo $wav | awk '{split($0,a,"_"); print a[1]}'`
        utt1=`echo $wav | awk '{split($0,a,"_"); print a[2]}'`
        snr1=`echo $wav | awk '{split($0,a,"_"); print a[3]}'`
        speaker2=`echo $wav | awk '{split($0,a,"_"); print a[4]}'`
        utt2=`echo $wav | awk '{split($0,a,"_"); print a[5]}'`

        wav_base=`echo $wav | sed 's|.wav||g'`

        mix_wav=$mix_path/$wav
        s1_wav=$s1_path/$wav_base+${speaker1}_$utt1.wav
        s2_wav=$s2_path/$wav_base+${speaker2}_$utt2.wav

        face1=$faces_data_folder/$kind/$speaker1/$utt1.npy 
        face2=$faces_data_folder/$kind/$speaker2/$utt2.npy 
        lip1=$lips_data_folder/$kind/$speaker1/$utt1.npy
        lip2=$lips_data_folder/$kind/$speaker2/$utt2.npy

        s1_cutted_wav=$cutted_audio_folder/$kind/$speaker1/$utt1.wav 
        s2_cutted_wav=$cutted_audio_folder/$kind/$speaker2/$utt2.wav 


        echo $mix_wav $s1_wav $s2_wav $face1 $face2 $lip1 $lip2 >> ./diff/$kind.scp
    done 
done 


for kind in `ls $audio_data_folder_s`
do 
    mix_path=$audio_data_folder_s/$kind/mix
    s1_path=$audio_data_folder_s/$kind/s1 
    s2_path=$audio_data_folder_s/$kind/s2 

    for wav in `ls $mix_path`
    do 

        speaker1=`echo $wav | awk '{split($0,a,"_"); print a[1]}'`
        utt1=`echo $wav | awk '{split($0,a,"_"); print a[2]}'`
        snr1=`echo $wav | awk '{split($0,a,"_"); print a[3]}'`
        speaker2=`echo $wav | awk '{split($0,a,"_"); print a[4]}'`
        utt2=`echo $wav | awk '{split($0,a,"_"); print a[5]}'`

        wav_base=`echo $wav | sed 's|.wav||g'`

        mix_wav=$mix_path/$wav
        s1_wav=$s1_path/$wav_base+${speaker1}_$utt1.wav
        s2_wav=$s2_path/$wav_base+${speaker2}_$utt2.wav

        face1=$faces_data_folder/$kind/$speaker1/$utt1.npy 
        face2=$faces_data_folder/$kind/$speaker2/$utt2.npy 
        lip1=$lips_data_folder/$kind/$speaker1/$utt1.npy
        lip2=$lips_data_folder/$kind/$speaker2/$utt2.npy

        s1_cutted_wav=$cutted_audio_folder/$kind/$speaker1/$utt1.wav 
        s2_cutted_wav=$cutted_audio_folder/$kind/$speaker2/$utt2.wav 

        echo $mix_wav $s1_wav $s2_wav $face1 $face2 $lip1 $lip2 >> ./same/$kind.scp
    done 
done 

