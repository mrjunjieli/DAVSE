# split test_scp into two parts: samegender mixture and diffgender mixture
#according to gender.scp 
import os 
gender_scp={}
with open('../gender.scp','r') as p:
    for line in p.readlines():
        name,gender = line.strip().split(' ')
        gender_scp[name] = gender

with open('./test.scp','r') as p:
    for line in p.readlines():
        print(line)
        mix_wav_path, s1_wav_path, s2_wav_path,face1_path, face2_path, lip1_path, lip2_path =line.strip().split(' ')
        s1_sp = lip1_path.split('/')[-2]
        s2_sp = lip2_path.split('/')[-2]

        if gender_scp[s1_sp] != gender_scp[s2_sp]:
            with open('./test_diffGender.scp','a+') as p:
                p.write(line)
        else:
            with open('./test_sameGender.scp','a+') as p:
                p.write(line)

