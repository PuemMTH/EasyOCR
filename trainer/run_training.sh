#!/bin/bash
# EasyOCR Thai Training - One Command Setup
# การฝึกโมเดล EasyOCR ภาษาไทยด้วยคำสั่งเดียว

set -e  # หยุดเมื่อเกิดข้อผิดพลาด

echo "🚀 EasyOCR Thai Training - One Command Setup"
echo "=============================================="

# ตรวจสอบว่าอยู่ใน trainer directory
if [ ! -f "pipeline.py" ]; then
    echo "❌ กรุณารันจาก trainer directory"
    echo "cd trainer && ./run_training.sh"
    exit 1
fi

# ตรวจสอบข้อมูล
if [ ! -d "all_data/results_JS-Kobori" ]; then
    echo "❌ ไม่พบข้อมูล: all_data/results_JS-Kobori/"
    echo "กรุณาวางข้อมูลในโฟลเดอร์นี้ก่อน"
    exit 1
fi

echo "📋 Step 1/3: รัน Pipeline"
uv run python pipeline.py --model thai_auto

echo ""
echo "📋 Step 2/3: เริ่มการฝึก"
echo "🔄 การฝึกจะใช้เวลาประมาณ 1-2 ชั่วโมง..."

# รันการฝึกและบันทึก log
uv run python quick_train.py 2>&1 | tee training.log

echo ""
echo "📋 Step 3/3: ตรวจสอบผลลัพธ์"

# ตรวจสอบว่าโมเดลถูกสร้างแล้ว
if [ -f "saved_models/thai_auto/best_accuracy.pth" ]; then
    echo "✅ การฝึกเสร็จสิ้น!"
    echo "📁 โมเดล: saved_models/thai_auto/best_accuracy.pth"
    
    # แสดงสถิติล่าสุด
    if [ -f "saved_models/thai_auto/log_train.txt" ]; then
        echo ""
        echo "📊 สถิติล่าสุด:"
        tail -n 10 saved_models/thai_auto/log_train.txt | grep -E "(Train loss|Current_accuracy)" || echo "ไม่พบข้อมูลสถิติ"
    fi
    
    echo ""
    echo "🎯 การใช้งานโมเดล:"
    echo "1. โมเดลอยู่ใน: saved_models/thai_auto/"
    echo "2. Config อยู่ใน: config_files/thai_auto_config.yaml"
    echo "3. Log อยู่ใน: saved_models/thai_auto/log_train.txt"
    
else
    echo "❌ การฝึกล้มเหลว!"
    echo "📋 ตรวจสอบ log:"
    
    if [ -f "training.log" ]; then
        echo "--- Last 20 lines of training.log ---"
        tail -n 20 training.log
    fi
    
    if [ -f "saved_models/thai_auto/log_train.txt" ]; then
        echo "--- Last 10 lines of log_train.txt ---"
        tail -n 10 saved_models/thai_auto/log_train.txt
    fi
    
    exit 1
fi

echo ""
echo "🎉 การฝึกโมเดล EasyOCR ภาษาไทยเสร็จสิ้น!"
