#!/usr/bin/env python3
"""
EasyOCR Thai Training Pipeline - Essential Steps Only
Pipeline การฝึกโมเดล EasyOCR ภาษาไทย (เฉพาะขั้นตอนจำเป็น)
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import csv

class EasyOCRPipeline:
    def __init__(self, data_dir="all_data/results_JS-Kobori", model_name="thai_model"):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.train_dir = Path("all_data/thai_train")
        self.config_dir = Path("config_files")
        self.saved_dir = Path(f"saved_models/{model_name}")
        
    def step1_check_data(self):
        """Step 1: ตรวจสอบข้อมูล"""
        print("📋 Step 1: ตรวจสอบข้อมูล")
        
        # ตรวจสอบโฟลเดอร์ข้อมูล
        if not self.data_dir.exists():
            raise FileNotFoundError(f"❌ ไม่พบโฟลเดอร์ข้อมูล: {self.data_dir}")
        
        # ตรวจสอบไฟล์ gt.txt
        gt_file = self.data_dir / "gt.txt"
        if not gt_file.exists():
            raise FileNotFoundError(f"❌ ไม่พบไฟล์ gt.txt: {gt_file}")
        
        # นับข้อมูล
        with open(gt_file, 'r', encoding='utf-8') as f:
            data_count = len(f.readlines())
        
        print(f"✅ พบข้อมูล: {data_count} ตัวอย่าง")
        return data_count
    
    def step2_prepare_data(self):
        """Step 2: เตรียมข้อมูล"""
        print("📊 Step 2: เตรียมข้อมูล")
        
        # สร้างโฟลเดอร์
        self.train_dir.mkdir(parents=True, exist_ok=True)
        
        # อ่าน gt.txt
        gt_file = self.data_dir / "gt.txt"
        labels_data = []
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        image_path = parts[0].replace('\\', '/')
                        text = parts[1]
                        filename = os.path.basename(image_path)
                        labels_data.append([filename, text])
        
        # สร้าง labels.csv
        labels_csv = self.train_dir / 'labels.csv'
        with open(labels_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'words'])
            writer.writerows(labels_data)
        
        # คัดลอกรูปภาพ
        source_images_dir = self.data_dir / "images" / "0"
        if source_images_dir.exists():
            copied_count = 0
            for filename in os.listdir(source_images_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    source_path = source_images_dir / filename
                    target_path = self.train_dir / filename
                    shutil.copy2(source_path, target_path)
                    copied_count += 1
            print(f"✅ คัดลอกรูป: {copied_count} ไฟล์")
        
        print(f"✅ สร้าง labels.csv: {len(labels_data)} แถว")
        return labels_data
    
    def step3_analyze_parameters(self, labels_data):
        """Step 3: วิเคราะห์ parameters (เฉพาะที่จำเป็น)"""
        print("⚙️ Step 3: วิเคราะห์ parameters")
        
        # วิเคราะห์ความยาวข้อความ
        text_lengths = [len(item[1]) for item in labels_data]
        max_length = max(text_lengths)
        recommended_max_length = min(max_length + 5, 50)  # จำกัดไม่เกิน 50
        
        # วิเคราะห์ตัวอักษร
        all_text = ''.join([item[1] for item in labels_data])
        unique_chars = sorted(set(all_text))
        
        print(f"✅ ตัวอักษร: {len(unique_chars)} ตัว")
        print(f"✅ ความยาวสูงสุด: {max_length} -> แนะนำ: {recommended_max_length}")
        
        return {
            'batch_max_length': recommended_max_length,
            'lang_char': ''.join(unique_chars),
            'num_chars': len(unique_chars)
        }
    
    def step4_create_config(self, params):
        """Step 4: สร้าง config file"""
        print("📝 Step 4: สร้าง config")
        
        self.config_dir.mkdir(exist_ok=True)
        
        config = f"""# Essential Thai OCR Config
number: '0123456789'
symbol: "!\\"#$%&'()*+,-./:;<=>?@[\\\\]^_`{{|}}~ €"
lang_char: '{params['lang_char']}'
experiment_name: '{self.model_name}'
train_data: 'all_data'
valid_data: 'all_data/thai_train'
manualSeed: 1111
workers: 0
batch_size: 8
num_iter: 5000
valInterval: 250
saved_model: ''
FT: False
optim: 'adam'
lr: 0.001
beta1: 0.9
rho: 0.95
eps: 0.00000001
grad_clip: 5
select_data: 'thai_train'
batch_ratio: '1'
total_data_usage_ratio: 1.0
batch_max_length: {params['batch_max_length']}
imgH: 64
imgW: 400
rgb: False
contrast_adjust: 0.0
sensitive: True
PAD: True
data_filtering_off: False
Transformation: 'None'
FeatureExtraction: 'VGG'
SequenceModeling: 'BiLSTM'
Prediction: 'CTC'
num_fiducial: 20
input_channel: 1
output_channel: 256
hidden_size: 256
decode: 'greedy'
new_prediction: False
freeze_FeatureFxtraction: False
freeze_SequenceModeling: False"""
        
        config_file = self.config_dir / f"{self.model_name}_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config)
        
        print(f"✅ สร้าง config: {config_file}")
        return config_file
    
    def step5_install_dependencies(self):
        """Step 5: ติดตั้ง dependencies (เฉพาะที่จำเป็น)"""
        print("📦 Step 5: ติดตั้ง dependencies")
        
        essential_packages = [
            "torch", "torchvision", "torchaudio",
            "opencv-python", "pillow", "numpy",
            "pyyaml", "pandas", "easydict"
        ]
        
        try:
            cmd = ["uv", "add"] + essential_packages
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ ติดตั้งแล้ว: {', '.join(essential_packages)}")
        except subprocess.CalledProcessError as e:
            print(f"❌ ติดตั้งล้มเหลว: {e}")
            return False
        
        return True
    
    def step6_start_training(self, config_file):
        """Step 6: เริ่มการฝึก"""
        print("🚀 Step 6: เริ่มการฝึก")
        
        # สร้าง training script แบบง่าย
        train_script = f"""
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
    os.makedirs(f'./saved_models/{{opt.experiment_name}}', exist_ok=True)
    return opt

if __name__ == '__main__':
    print("Loading config from: {config_file}")
    opt = get_config("{config_file}")
    print(f"Starting training: {{opt.experiment_name}}")
    print(f"Character set size: {{len(opt.character)}}")
    
    try:
        train(opt, amp=False)
    except Exception as e:
        print(f"Training error: {{e}}")
        print("Check log files in saved_models/ directory")
"""
        
        train_file = Path("quick_train.py")
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write(train_script)
        
        print(f"✅ สร้าง training script: {train_file}")
        print("💡 รันการฝึกด้วย: uv run python quick_train.py")
        
        return train_file
    
    def run_pipeline(self):
        """รัน pipeline ทั้งหมด"""
        print("🔄 EasyOCR Thai Training Pipeline")
        print("=" * 50)
        
        try:
            # Step 1: ตรวจสอบข้อมูล
            data_count = self.step1_check_data()
            
            # Step 2: เตรียมข้อมูล
            labels_data = self.step2_prepare_data()
            
            # Step 3: วิเคราะห์ parameters
            params = self.step3_analyze_parameters(labels_data)
            
            # Step 4: สร้าง config
            config_file = self.step4_create_config(params)
            
            # Step 5: ติดตั้ง dependencies
            if not self.step5_install_dependencies():
                print("❌ ติดตั้ง dependencies ล้มเหลว")
                return False
            
            # Step 6: เตรียมการฝึก
            train_file = self.step6_start_training(config_file)
            
            print("\n" + "=" * 50)
            print("✅ Pipeline เสร็จสิ้น!")
            print(f"📊 ข้อมูล: {data_count} ตัวอย่าง")
            print(f"🔤 ตัวอักษร: {params['num_chars']} ตัว")
            print(f"📏 ความยาวสูงสุด: {params['batch_max_length']}")
            print(f"⚙️ Config: {config_file}")
            print(f"🚀 Training script: {train_file}")
            
            print("\n🎯 ขั้นตอนต่อไป:")
            print("1. รันการฝึก: uv run python quick_train.py")
            print("2. ดู log ใน: saved_models/thai_model/")
            print("3. โมเดลจะบันทึกใน: saved_models/thai_model/best_accuracy.pth")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline ล้มเหลว: {e}")
            return False

def main():
    """ฟังก์ชันหลัก"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EasyOCR Thai Training Pipeline")
    parser.add_argument("--data", default="all_data/results_JS-Kobori", help="โฟลเดอร์ข้อมูล")
    parser.add_argument("--model", default="thai_model", help="ชื่อโมเดล")
    
    args = parser.parse_args()
    
    pipeline = EasyOCRPipeline(args.data, args.model)
    success = pipeline.run_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
