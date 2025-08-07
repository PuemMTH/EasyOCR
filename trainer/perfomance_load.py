import typer
import os
import time
import csv
import psutil
import subprocess
from datetime import datetime
import sys
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd
from rich.console import Console

app = typer.Typer()

def long_time_process():
    time.sleep(20)

def get_gpu_info():
    """Get NVIDIA GPU information using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_data = []
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 8:
                        gpu_data.append({
                            'gpu_index': parts[0],
                            'gpu_name': parts[1],
                            'memory_total_mb': parts[2],
                            'memory_used_mb': parts[3],
                            'memory_free_mb': parts[4],
                            'gpu_utilization_percent': parts[5],
                            'memory_utilization_percent': parts[6],
                            'temperature_celsius': parts[7]
                        })
            return gpu_data
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        pass
    return []

def get_cpu_info():
    """Get CPU information"""
    cpu_count = psutil.cpu_count()
    cpu_percent_per_core = psutil.cpu_percent(interval=1, percpu=True)
    cpu_percent_total = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        'cpu_count': cpu_count,
        'cpu_percent_total': cpu_percent_total,
        'cpu_percent_per_core': cpu_percent_per_core,
        'memory_total_gb': memory.total / (1024**3),
        'memory_used_gb': memory.used / (1024**3),
        'memory_percent': memory.percent
    }

def monitor_performance(output_dir: str, interval: int = 5):
    """Monitor performance metrics and save to CSV"""
    gpu_csv_path = os.path.join(output_dir, "gpu_performance.csv")
    cpu_csv_path = os.path.join(output_dir, "cpu_performance.csv")
    
    # Create CSV files with headers
    gpu_headers = ['timestamp', 'gpu_index', 'gpu_name', 'memory_total_mb', 'memory_used_mb', 
                   'memory_free_mb', 'gpu_utilization_percent', 'memory_utilization_percent', 'temperature_celsius']
    cpu_headers = ['timestamp', 'cpu_count', 'cpu_percent_total', 'memory_total_gb', 
                   'memory_used_gb', 'memory_percent'] + [f'cpu_core_{i}' for i in range(psutil.cpu_count())]
    
    # Initialize CSV files
    with open(gpu_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(gpu_headers)
    
    with open(cpu_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cpu_headers)
    
    print(f"Monitoring started. GPU data: {gpu_csv_path}")
    print(f"CPU data: {cpu_csv_path}")
    print(f"Monitoring interval: {interval} seconds")
    print("Press Ctrl+C to stop monitoring...")
    
    try:
        while True:
            timestamp = datetime.now().isoformat()
            
            # Get GPU data
            gpu_data = get_gpu_info()
            with open(gpu_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for gpu in gpu_data:
                    writer.writerow([
                        timestamp,
                        gpu['gpu_index'],
                        gpu['gpu_name'],
                        gpu['memory_total_mb'],
                        gpu['memory_used_mb'],
                        gpu['memory_free_mb'],
                        gpu['gpu_utilization_percent'],
                        gpu['memory_utilization_percent'],
                        gpu['temperature_celsius']
                    ])
            
            # Get CPU data
            cpu_data = get_cpu_info()
            with open(cpu_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [
                    timestamp,
                    cpu_data['cpu_count'],
                    cpu_data['cpu_percent_total'],
                    cpu_data['memory_total_gb'],
                    cpu_data['memory_used_gb'],
                    cpu_data['memory_percent']
                ] + cpu_data['cpu_percent_per_core']
                writer.writerow(row)
            
            # Print current status
            print(f"[{timestamp}] GPU: {len(gpu_data)} devices, CPU: {cpu_data['cpu_percent_total']:.1f}%, Memory: {cpu_data['memory_percent']:.1f}%")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"Error during monitoring: {e}")

@app.command()
def perfomance_load(
        output_path: str = typer.Option(..., "--output", "-o", help="The name of the output."),
    ):
    """
    Run the performance load.
    """
    start_time = time.time()
    typer.echo("Running performance load...")
    long_time_process()
    end_time = time.time()
    typer.echo(f"Time taken: {end_time - start_time} seconds")
    output_dir = os.path.join(".", output_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "output.txt"), "w") as f:
        f.write(f"Time taken: {end_time - start_time} seconds")
    typer.echo(f"Output saved to {output_dir}")

@app.command()
def monitor(
    output_dir: str = typer.Option("performance_logs", "--output-dir", "-o", help="Output directory for CSV files"),
    interval: int = typer.Option(1, "--interval", "-i", help="Monitoring interval in seconds"),
    background: bool = typer.Option(False, "--background", "-b", help="Run in background mode")
):
    """
    Monitor GPU and CPU performance metrics and save to CSV files.
    
    GPU metrics: GPU utilization, memory usage, temperature
    CPU metrics: Core count, CPU load per core, total CPU load, memory usage
    
    example:
    python perfomance_load.py monitor --output-dir performance_logs --interval 1 --background
    python perfomance_load.py monitor -o performance_logs -i 1 -b
    """
    os.makedirs(output_dir, exist_ok=True)
    # clear the output directory
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))
    
    if background:
        print(f"Starting background monitoring to {output_dir}")
        print("Logs will be saved to:")
        print(f"  GPU: {os.path.join(output_dir, 'gpu_performance.csv')}")
        print(f"  CPU: {os.path.join(output_dir, 'cpu_performance.csv')}")
        print(f"Monitoring interval: {interval} seconds")
        print("Use 'ps aux | grep perfomance_load' to find the process")
        print("Use 'kill <pid>' to stop monitoring")
        monitor_performance(output_dir, interval)
    else:
        monitor_performance(output_dir, interval)

@app.command()
def training(
    config_file: str = typer.Option(..., "--config", "-c", help="The path to the config file (required)"),
    experiment_name: str = typer.Option(..., "--experiment-name", "-e", help="The name of the experiment (required)"),
    training_data: str = typer.Option(..., "--train-data", "-t", help="The path to the train data (required)"),
    amp: bool = typer.Option(False, "--amp", "-a", help="Use automatic mixed precision training"),
):
    """
    Run training using the specified config file.

    example:
    python perfomance_load.py training --config config_files/th_filtered_config.yaml --experiment-name th_filtered_2gpus --train-data all_data/thai_easyocr_format --amp
    python perfomance_load.py training -c config_files/th_filtered_config.yaml -e th_filtered_2gpus -t all_data/thai_easyocr_format -a
    """
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

    opt = get_config(config_file)
    console = Console()

    opt.experiment_name = experiment_name
    opt.train_data = training_data
    opt.valid_data = os.path.join(training_data, "val")
    console.print(opt)
    train(opt, amp=amp)

if __name__ == "__main__":
    app()