import os
import torch
import argparse
import subprocess

assert torch.cuda.is_available(), "\033[31m You need GPU to Train! \033[0m"
print("CPU Count is :", os.cpu_count())

# filenames = []
# src_dir = '/home/yunwei/new/VBeautifier/test_audio/yt_singing'
# # src_dir = '/home/yunwei/new/VBeautifier/test_audio/train'
# for filename in os.listdir(src_dir):
#     filenames.append(os.path.join(src_dir, filename))

commands = []
stylename = '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/yw/beipan.wav'
# stylename = '/home/yunwei/new/voice_synthesis/whisper-vits-svc/beipan_svc.wav'
# stylename = '/root/vbeaut/OpenSinger/Woman/waves-32k/42_爱笑的眼睛/42_爱笑的眼睛_4.wav'
# stylename = '/root/vbeaut/OpenSinger/Woman/waves-32k/42_情深深雨濛濛/42_情深深雨濛濛_5.wav'
# filenames = ['/root/vbeaut/OpenSinger/Woman/waves-32k/42_情深深雨濛濛/42_情深深雨濛濛_5.wav']
# filenames = ['/root/vbeaut/OpenSinger/WomanRaw/0_父亲写的散文诗/0_父亲写的散文诗_4.wav']
# stylename = '/root/vbeaut/OpenSinger/WomanRaw/27_父亲写的散文诗/27_父亲写的散文诗_4.wav'

# filenames = ['/root/vbeaut/OpenSinger/WomanRaw/40_天黑黑/40_天黑黑_0.wav']
# stylename = '/root/vbeaut/OpenSinger/WomanRaw/40_天黑黑/40_天黑黑_0.wav'
# filenames = ['/root/vbeaut/OpenSinger/WomanRaw/6_修炼爱情/6_修炼爱情_15.wav']
# stylename = '/root/vbeaut/OpenSinger/WomanRaw/17_修炼爱情/17_修炼爱情_15.wav'
# stylename = '/root/vbeaut/OpenSinger/Woman/waves-32k/20_手心的蔷薇/20_手心的蔷薇_3.wav'
# filenames = ['/root/vbeaut/OpenSinger/Woman/waves-32k/42_爱笑的眼睛/42_爱笑的眼睛_4.wav']
# filenames = ['/root/vbeaut/OpenSinger/Woman/waves-32k/20_手心的蔷薇/20_手心的蔷薇_3.wav']
# stylename = '/root/vbeaut/VBeautifier/liujiahui_yongyou.wav'
# filenames = ['/root/vbeaut/VBeautifier/wholeNewWorldOrig.wav']
# stylename = '/root/vbeaut/VBeautifier/zuichu_orig.wav'
# filenames = ['/root/vbeaut/VBeautifier/zuichu_orig.wav']#, \
# '/root/vbeaut/VBeautifier/beipan_orig_clipped.wav',\
# '/root/vbeaut/VBeautifier/wholeNewWorldOrig.wav']
# stylename = '/root/vbeaut/VBeautifier/liangxinyi_xiayutian_v.wav'
# stylename = '/root/vbeaut/VBeautifier/wholeNewWorldOrig.wav'
# stylename = 'test_audio/male_style/11_那些你很冒险的梦_17.wav'
# stylename = '/home/yunwei/new/VBeautifier/test_audio/userstudy/liangcheng/lemon_ugly.wav'
# filenames = ['/home/yunwei/new/VBeautifier/test_audio/userstudy/liangcheng/lemon_ugly.wav']
# stylename = 'test.wav'
# stylename = '/home/yunwei/new/VBeautifier/test_audio/male_style/18_花海_8.wav'
# stylename = '/mnt/hdd/yw/VBeautifier/overall_data/dataset_fun/labmate/don_t-look-back-in-anger.wav'
# filenames = ['test_audio/male_style/11_那些你很冒险的梦_17.wav']
# filenames = ['/home/yunwei/new/VBeautifier/test_audio/male_style/18_花海_8.wav']
# filenames = ['test.wav']
# filenames = ['test_audio/train/42_情深深雨濛濛_5.wav']
filenames = [
    # '/mnt/hdd/yw/VBeautifier/OpenSinger/Woman/waves-32k/26_修炼爱情/26_修炼爱情_14.wav',\
    # '/mnt/hdd/yw/VBeautifier/OpenSinger/Woman/waves-32k/6_修炼爱情/6_修炼爱情_10.wav',\
    # 'test_audio/train/42_情深深雨濛濛_5.wav',\
    # '/mnt/hdd/yw/VBeautifier/OpenSinger/Woman/waves-32k/42_爱笑的眼睛/42_爱笑的眼睛_6.wav',\
    # '/mnt/hdd/yw/VBeautifier/OpenSinger/Woman/waves-32k/27_父亲写的散文诗/27_父亲写的散文诗_6.wav',\
    # '/mnt/hdd/yw/VBeautifier/OpenSinger/Woman/waves-32k/0_父亲写的散文诗/0_父亲写的散文诗_6.wav',\
    # '/mnt/hdd/yw/VBeautifier/OpenSinger/Woman/waves-32k/20_手心的蔷薇/20_手心的蔷薇_6.wav',\
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/yw/21gunsUnnatural.wav',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/yw/beipan_long.wav',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/yw/happyEndingOrig.wav',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/yw/hongdou_clipped.wav',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/yw/kongBaiGeSelf.wav',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/yw/liuNian.wav',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/yw/reflection_clipped.wav',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/yw/最初的梦想_clipped.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/yt_singing/letItGoMale.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/xiangming/you_raise_me_up_ugly.wav',
    '/home/yunwei/new/VBeautifier/test_audio/yt_singing/rolling.m4a'
]
spks = [
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/21gunsUnnatural.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/beipanlong.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/happyEndingOrig.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/hongdou_clipped.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/kongBaiGeSelf.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/liuNian.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/reflection_clipped.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/最初的梦想_clipped.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/yt/letItGoMale.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/xm/you_raise_me_up_ugly.spk.npy',
    '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/yt/someone.spk.npy'
]
for i, filename in enumerate(filenames):
    # stylename, filename = filename, stylename
    # stylename = filename
    # command1 = f'python voicep_inference.py --config configs/beauty.yaml --wave {filename} --style_wave {filename}'
    spk = spks[i]
    command = f'python svc_inference.py --config configs/base.yaml --model /home/yunwei/new/voice_synthesis/whisper-vits-svc/vits_pretrain/sovits5.0.pretrain.pth --spk {spk} --wave {filename}'
    commands.extend([command])
    # break
    # commands.extend([command1, command2])

for command in commands:
   print(f"Command: {command}")

   process = subprocess.Popen(command, shell=True)
   outcode = process.wait()
   if (outcode):
      break
