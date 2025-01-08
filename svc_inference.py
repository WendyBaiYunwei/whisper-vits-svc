import logging
import sys,os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import argparse
import numpy as np
from scipy.io import wavfile
import librosa

from omegaconf import OmegaConf
from scipy.io.wavfile import write
from vits.models import SynthesizerInfer
from vits import utils, spectrogram
from pitch import load_csv_pitch
from pydub import AudioSegment
# from feature_retrieval import IRetrieval, DummyRetrieval, FaissIndexRetrieval, load_retrieve_index

logger = logging.getLogger(__name__)


def get_speaker_name_from_path(speaker_path: Path) -> str:
    suffixes = "".join(speaker_path.suffixes)
    filename = speaker_path.name
    return filename.rstrip(suffixes)


# def create_retrival(cli_args) -> IRetrieval:
#     if not cli_args.enable_retrieval:
#         logger.info("infer without retrival")
#         return DummyRetrieval()
#     else:
#         logger.info("load index retrival model")

#     speaker_name = get_speaker_name_from_path(Path(args.spk))
#     base_path = Path(".").absolute() / "data_svc" / "indexes" / speaker_name

#     if cli_args.hubert_index_path:
#         hubert_index_filepath = cli_args.hubert_index_path
#     else:
#         index_name = f"{cli_args.retrieval_index_prefix}hubert.index"
#         hubert_index_filepath = base_path / index_name

#     if cli_args.whisper_index_path:
#         whisper_index_filepath = cli_args.whisper_index_path
#     else:
#         index_name = f"{cli_args.retrieval_index_prefix}whisper.index"
#         whisper_index_filepath = base_path / index_name

#     return FaissIndexRetrieval(
#         hubert_index=load_retrieve_index(
#             filepath=hubert_index_filepath,
#             ratio=cli_args.retrieval_ratio,
#             n_nearest_vectors=cli_args.n_retrieval_vectors
#         ),
#         whisper_index=load_retrieve_index(
#             filepath=whisper_index_filepath,
#             ratio=cli_args.retrieval_ratio,
#             n_nearest_vectors=cli_args.n_retrieval_vectors
#         ),
#     )


def load_svc_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model_g"]
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


def svc_infer(model, retrieval, spk, pit, ppg, vec, hp, device, spec):
    len_pit = pit.size()[0]
    len_vec = vec.size()[0]
    len_ppg = ppg.size()[0]
    len_spec = spec.size()[-1]
    len_min = min(len_pit, len_vec)
    len_min = min(len_min, len_ppg)
    len_min = min(len_min, len_spec)
    pit = pit[:len_min]
    vec = vec[:len_min, :]
    ppg = ppg[:len_min, :]
    spec = spec[:, :, :len_min]

    with torch.no_grad():
        spk = spk.unsqueeze(0).to(device)
        source = pit.unsqueeze(0).to(device)
        source = model.pitch2source(source)
        pitwav = model.source2wav(source)
        write("svc_out_pit.wav", hp.data.sampling_rate, pitwav)

        hop_size = hp.data.hop_length
        all_frame = len_min
        hop_frame = 10
        out_chunk = 2500  # 25 S
        out_index = 0
        out_audio = []

        while (out_index < all_frame):

            if (out_index == 0):  # start frame
                cut_s = 0
                cut_s_out = 0
            else:
                cut_s = out_index - hop_frame
                cut_s_out = hop_frame * hop_size

            if (out_index + out_chunk + hop_frame > all_frame):  # end frame
                cut_e = all_frame
                cut_e_out = -1
            else:
                cut_e = out_index + out_chunk + hop_frame
                cut_e_out = -1 * hop_frame * hop_size

            sub_ppg = ppg[cut_s:cut_e, :]
            sub_vec = vec[cut_s:cut_e, :]
            sub_ppg = sub_ppg.unsqueeze(0).to(device)
            sub_vec = sub_vec.unsqueeze(0).to(device)
            sub_pit = pit[cut_s:cut_e].unsqueeze(0).to(device)
            # print(pit.shape)
            # print(spec.shape)
            # exit()
            sub_spec = spec[:, :, cut_s:cut_e].to(device)
            sub_len = torch.LongTensor([cut_e - cut_s]).to(device)
            sub_har = source[:, :, cut_s *
                             hop_size:cut_e * hop_size].to(device)
            if args.mode == 'baseline':
                sub_out = model.baseline(
                    sub_ppg, sub_vec, sub_pit, spk, sub_len, sub_har, sub_spec)
            elif args.mode == 'beaut_gauss':
                sub_out = model.inference(
                    sub_ppg, sub_vec, sub_pit, spk, sub_len, sub_har, sub_spec, args.extent, gaussian=True)
            elif args.mode == 'beaut':
                sub_out = model.inference(
                    sub_ppg, sub_vec, sub_pit, spk, sub_len, sub_har, sub_spec, args.extent)
            elif args.mode == 'no_pca':
                sub_out = model.no_pca_inference(
                    sub_ppg, sub_vec, sub_pit, spk, sub_len, sub_har, sub_spec)
            sub_out = sub_out[0, 0].data.cpu().detach().numpy()

            sub_out = sub_out[cut_s_out:cut_e_out]
            out_audio.extend(sub_out)
            out_index = out_index + out_chunk

        out_audio = np.asarray(out_audio)
    return out_audio


def main(args):
    hp = OmegaConf.load(args.config)
    if '.m4a' in args.wave:
        audio = AudioSegment.from_file(args.wave, format="m4a")
        args.wave = args.wave.replace('m4a', 'wav')
        audio.export(args.wave, format="wav")
    audio, sampling_rate = utils.load_wav_to_torch(args.wave)
    if sampling_rate != hp.data.sampling_rate:
        wav, _ = librosa.load(args.wave, sr=hp.data.sampling_rate)
        wav = wav / np.abs(wav).max() * 0.6
        wav = wav / max(0.01, np.max(np.abs(wav))) * 32767 * 0.6
        wavfile.write(args.wave, hp.data.sampling_rate, wav.astype(np.int16))
        audio, sampling_rate = utils.load_wav_to_torch(args.wave)
    assert sampling_rate == hp.data.sampling_rate
    audio_norm = audio / hp.data.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    n_fft = hp.data.filter_length
    sampling_rate = hp.data.sampling_rate
    hop_size = hp.data.hop_length
    win_size = hp.data.win_length
    spec = spectrogram.spectrogram_torch(
        audio_norm, n_fft, sampling_rate, hop_size, win_size, center=False)

    if (args.ppg == None):
        args.ppg = "svc_tmp.ppg.npy"
        print(
            f"Auto run : python whisper/inference.py -w {args.wave} -p {args.ppg}")
        os.system(f"python whisper/inference.py -w {args.wave} -p {args.ppg}")

    if (args.vec == None):
        args.vec = "svc_tmp.vec.npy"
        print(
            f"Auto run : python hubert/inference.py -w {args.wave} -v {args.vec}")
        os.system(f"python hubert/inference.py -w {args.wave} -v {args.vec}")

    if (args.pit == None):
        args.pit = "svc_tmp.pit.csv"
        print(
            f"Auto run : python pitch/inference.py -w {args.wave} -p {args.pit}")
        os.system(f"python pitch/inference.py -w {args.wave} -p {args.pit}")

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)
    
    if args.model == None and args.mode == 'baseline':
        args.model = '/home/yunwei/new/voice_synthesis/whisper-vits-svc/vits_pretrain/sovits5.0.pretrain.pth'
    elif args.model == None and 'beaut' in args.mode:
        args.model = '/home/yunwei/new/voice_synthesis/whisper-vits-svc/chkpt/gauss_pca/gauss_pca1.pt'
    else:
        args.model = '/home/yunwei/new/voice_synthesis/whisper-vits-svc/chkpt/no_pca_train/no_pca_train1.pt'
    load_svc_model(args.model, model)
    # retrieval = create_retrival(args)
    model.eval()
    model.to(device)

    spk = np.load(args.spk)
    spk = torch.FloatTensor(spk)
    spk = torch.ones(spk.shape) * spk.mean()

    ppg = np.load(args.ppg)
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)
    # ppg = torch.zeros_like(ppg)

    vec = np.load(args.vec)
    vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
    vec = torch.FloatTensor(vec)
    # vec = torch.zeros_like(vec)

    pit = load_csv_pitch(args.pit)
    print("pitch shift: ", args.shift)
    if (args.shift == 0):
        pass
    else:
        pit = np.array(pit)
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        shift = args.shift
        shift = 2 ** (shift / 12)
        pit = pit * shift
    pit = torch.FloatTensor(pit)

    out_audio = svc_infer(model, None, spk, pit, ppg, vec, hp, device, spec)
    # wave_name = args.wave.replace('.wav', '_syn.wav')
    name = args.wave.split('/')[-1][:-len('.wav')]
    wave_name = f'{name}_{args.mode}_{args.note}.wav'
    write(wave_name, hp.data.sampling_rate, out_audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('--model', type=str, 
                        help="path of model for evaluation")
    parser.add_argument('--wave', type=str, required=True,
                        help="Path of raw audio.")
    parser.add_argument('--spk', type=str, required=True,
                        help="Path of speaker.")
    parser.add_argument('--ppg', type=str,
                        help="Path of content vector.")
    parser.add_argument('--vec', type=str,
                        help="Path of hubert vector.")
    parser.add_argument('--pit', type=str,
                        help="Path of pitch csv file.")
    parser.add_argument('--shift', type=int, default=0,
                        help="Pitch shift key.")
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'beaut_gauss', 'beaut', 'no_pca'])
    parser.add_argument('--extent', type=int, default=50)
    parser.add_argument('--note', type=str, default='')

    # parser.add_argument('--enable-retrieval', action="store_true",
    #                     help="Enable index feature retrieval")
    # parser.add_argument('--retrieval-index-prefix', default='',
    #                     help='retrieval index file prefix. Will load file %prefix%hubert.index/%prefix%whisper.index')
    # parser.add_argument('--retrieval-ratio', type=float, default=.5,
    #                     help="ratio of feature retrieval effect. Must be in range 0..1")
    # parser.add_argument('--n-retrieval-vectors', type=int, default=3,
    #                     help="get n nearest vectors from retrieval index. Works stably in range 1..3")
    # parser.add_argument('--hubert-index-path', required=False,
    #                     help='path to hubert index file. Default data_svc/indexes/speaker.../%prefix%hubert.index')
    # parser.add_argument('--whisper-index-path', required=False,
    #                     help='path to whisper index file. Default data_svc/indexes/speaker.../%prefix%whisper.index')

    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    main(args)
