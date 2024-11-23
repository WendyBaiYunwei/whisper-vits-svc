import os
import random

def print_error(info):
    print(f"\033[31m File isn't existed: {info}\033[0m")

IndexBySinger = False
if __name__ == "__main__":
    os.makedirs("./files/", exist_ok=True)

    rootPath = "./data_svc/waves-32k/"
    all_items = []
    for spks in os.listdir(f"./{rootPath}"):
        if not os.path.isdir(f"./{rootPath}/{spks}"):
            continue
        print(f"./{rootPath}/{spks}")
        for file in os.listdir(f"./{rootPath}/{spks}"):
            if file.endswith("_syn.wav"):
                file = file[:-len("_syn.wav")]

                path_wave = f"./data_svc/waves-32k/{spks}/{file}.wav"
                path_pair = f"./data_svc/waves-32k/{spks}/{file}_syn.wav"
                if not os.path.isfile(path_pair):
                    continue
                if not os.path.isfile(path_wave):
                    continue
                
                all_items.append(
                    f"{path_wave}|{path_pair}")

    random.shuffle(all_items)
    trains = all_items
    # trains.sort()
    fw = open("./files/train.txt", "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()
