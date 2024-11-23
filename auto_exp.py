import os
import torch
import argparse
import subprocess

MODEL = '/home/yunwei/new/VBeautifier/sovits5.0.pretrain.pth'
BASE = '/mnt/hdd/yw/VBeautifier/overall_data/data_svc_professional'

def rename(name):
    new_name = name.replace('\'', '')
    os.rename(name, new_name)
    return new_name

for singer in list(os.listdir(os.path.join(BASE, 'whisper'))):
    new_name = rename(os.path.join(BASE, 'waves-32k', singer))
    rename(os.path.join(BASE, 'whisper', singer))
    rename(os.path.join(BASE, 'hubert', singer))
    for filename in list(os.listdir(new_name)):
        if '_syn' in filename:
            continue
        filename = filename.replace('.wav', '')
        singer = singer.replace('\'', '')
        singer_filename = singer+'/'+filename
        wave = os.path.join(BASE, 'waves-32k', singer_filename+'.wav')
        ppg = os.path.join(BASE, 'whisper', singer_filename+'.ppg.npy')
        vec = os.path.join(BASE, 'hubert', singer_filename+'.vec.npy')
        rename(wave)
        rename(ppg)
        rename(vec)
        command = f"python svc_inference.py --config configs/base.yaml --model {MODEL} --wave {wave} --shift 0 --ppg {ppg} --vec {vec} --spk {BASE}/singer/Female1#singing#Almost_lover_Professional.spk.npy"
        print(f"Command: {command}")
        process = subprocess.Popen(command, shell=True)
        outcode = process.wait()
        if (outcode):
            break
