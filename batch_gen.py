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
# stylename = '/home/yunwei/new/voice_synthesis/whisper-vits-svc/spks/yw/beipan.wav'
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
    # 'test_audio/imagine_bad.wav',
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
    # '/home/yunwei/new/VBeautifier/test_audio/yt_singing/rolling.m4a'
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/yunwei/reflection_clipped2.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/yunwei/reflection_clipped.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/userstudy_round2/diff/最初的梦想_clipped.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/new_tests/you_raise_me_up_ugly.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/new_tests/love_story_ugly.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/new_tests/let_it_go_ugly.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/new_tests/Someone_bad.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/new_tests/Rolling_in_the_deep_1_bad.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/new_tests/Only_love_bad.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/new_tests/Need_you_now_bad.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/new_tests/Apologize_bad.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/liangcheng/lemon_ugly.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/liangcheng/heart_ugly2.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/liangcheng/heart_ugly.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/liangcheng/colors_ugly.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/stabilization/9_明天你好_10.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/demo/yw/theDayYouWentAwayUnnatural.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/demo/final/happyEndingOrig.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/demo/final/wholeNewWorldOrig.wav',
    # 'test_audio/心跳_bad.wav',
    # 'test_audio/imagine_bad.wav',
    # 'test_audio/如果爱忘了_bad.wav',
    # 'test_audio/heal_the_world_bad.wav',
    'test_audio/first_love_bad.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/new_tests/handInHand.m4a',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/new_tests/imagine.m4a',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/new_tests/lvGuang.m4a',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/userstudy_round2/archive/最初的梦想.m4a',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/userstudy_round2/raw/给我一个理由忘记.wav',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/userstudy_round2/raw/high/给未来的自己.m4a'
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/userstudy_round2/raw/high/hongdou.m4a'
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/nsvb_before1.wav',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/nsvb_before2.wav',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/nsvb_before3.wav',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/nsvb_before4.wav',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/nsvb_before5.wav',
    # '/home/yunwei/new/VBeautifierDemo/audio/you_raise_me_up_ugly.wav',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/test_audio/spk/female_train/mingtian.wav',
    # '/home/yunwei/new/VBeautifierDemo/audio/shiyi.wav'
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
    '/mnt/hdd/yw/VBeautifier/overall_data/data_svc2/singer/cpop_singer.spk.npy'
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/test_audio/spk1/nsvb_before1.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/test_audio/spk2/nsvb_before2.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/test_audio/spk3/nsvb_before3.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/test_audio/spk4/nsvb_before4.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/test_audio/spk5/nsvb_before5.spk.npy'
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/singer/xiangming.spk.npy',
    # '/home/yunwei/new/VBeautifier/test_audio/userstudy/singer/yangtao.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/test_audio/spk/female_train/mingtian.spk.npy',
    # '/home/yunwei/new/voice_synthesis/whisper-vits-svc/test_audio/spk/male_train/shiyi_beaut_gaussian.spk.npy'
]
for i, filename in enumerate(filenames):
    # if i == 0:
    #     continue
    # stylename, filename = filename, stylename
    # stylename = filename
    # command1 = f'python voicep_inference.py --config configs/beauty.yaml --wave {filename} --style_wave {filename}'
    spk = spks[0] ##spks[0]

    # command = f'python svc_inference.py --config configs/base.yaml --mode beaut --spk {spk} --wave {filename} --note acl'
    # commands.extend([command])

    command = f'python svc_inference.py --config configs/base.yaml --mode beaut_gauss --spk {spk} --wave {filename} --note acl'
    commands.extend([command])

    # command = f'python svc_inference.py --config configs/base.yaml --mode no_pca --spk {spk} --wave {filename} --note acl'
    # commands.extend([command])

    # command = f'python svc_inference.py --config configs/base.yaml --mode beaut_gauss --spk {spk} --wave {filename} --extent 25 --note acl'
    # commands.extend([command])

    # command = f'python svc_inference.py --config configs/base.yaml --mode beaut_gauss --spk {spk} --wave {filename} --extent 100 --note acl'
    # commands.extend([command])
    break
    # commands.extend([command1, command2])

for command in commands:
   print(f"Command: {command}")

   process = subprocess.Popen(command, shell=True)
   outcode = process.wait()
   if (outcode):
      break
