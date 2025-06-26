
import os
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict

cudnn.benchmark = True
cudnn.deterministic = False

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

if __name__ == '__main__':
    print("Loading config from: config_files/thai_model_config.yaml")
    opt = get_config("config_files/thai_model_config.yaml")
    print(f"Starting training: {opt.experiment_name}")
    print(f"Character set size: {len(opt.character)}")
    
    try:
        train(opt, amp=False)
    except Exception as e:
        print(f"Training error: {e}")
        print("Check log files in saved_models/ directory")
