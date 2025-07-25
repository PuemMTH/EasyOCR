import os
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd
import sys

def main():
    print("GPU list: ", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
    
    cudnn.benchmark = True
    cudnn.deterministic = False
    
    def get_config(file_path):
        with open(file_path, 'r', encoding="utf8") as stream:
            opt = yaml.safe_load(stream)
        opt = AttrDict(opt)
        if opt.lang_char == 'None':
            characters = ''
            for data in opt['select_data'].split('-'):
                csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
                df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
                all_char = ''.join(df['words'])
                characters += ''.join(set(all_char))
            characters = sorted(set(characters))
            opt.character= ''.join(characters)
        else:
            opt.character = opt.number + opt.symbol + opt.lang_char
        os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
        return opt
    
    # ใช้ config file เดียวกับใน notebook
    config_file = "config_files/th_filtered_config.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    opt = get_config(config_file)
    train(opt, amp=False)

if __name__ == '__main__':
    main()