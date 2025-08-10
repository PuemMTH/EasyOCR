import typer
import os
import time
import csv
import psutil
import subprocess
from datetime import datetime
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd
from rich.console import Console

app = typer.Typer()
console = Console()

def get_gpu_info():
    """Get NVIDIA GPU information using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=1)
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

def monitor_performance(output_dir: str, interval: int = 1):
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
                    writer.writerow([timestamp, gpu['gpu_index'], gpu['gpu_name'], gpu['memory_total_mb'], gpu['memory_used_mb'],
                                     gpu['memory_free_mb'], gpu['gpu_utilization_percent'], gpu['memory_utilization_percent'], gpu['temperature_celsius']])
            # Get CPU data
            cpu_data = get_cpu_info()
            with open(cpu_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, cpu_data['cpu_count'], cpu_data['cpu_percent_total'], cpu_data['memory_total_gb'],
                                 cpu_data['memory_used_gb'], cpu_data['memory_percent']] + cpu_data['cpu_percent_per_core'])
            
            # Get CPU data
            cpu_data = get_cpu_info()
            with open(cpu_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [timestamp, cpu_data['cpu_count'], cpu_data['cpu_percent_total'], cpu_data['memory_total_gb'], cpu_data['memory_used_gb'],
                       cpu_data['memory_percent']] + cpu_data['cpu_percent_per_core']
                writer.writerow(row)
            
            # Print current status
            print(f"[{timestamp}] GPU: {len(gpu_data)} devices, CPU: {cpu_data['cpu_percent_total']:.1f}%, Memory: {cpu_data['memory_percent']:.1f}%")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"Error during monitoring: {e}")

@app.command()
def monitor(
    folder_name: str = typer.Option("performance_logs", "--output-name", "-o", help="Output folder name"),
    interval: int = typer.Option(1, "--interval", "-i", help="Monitoring interval in seconds"),
):
    """
    Monitor GPU and CPU performance metrics and save to CSV files.
    
    GPU metrics: GPU utilization, memory usage, temperature
    CPU metrics: Core count, CPU load per core, total CPU load, memory usage
    
    example:
    python perfomance_load.py monitor --output-dir performance_logs --interval 1
    python perfomance_load.py monitor -o performance_logs -i 1
    """

    os.makedirs(f"./performance_logs/{folder_name}", exist_ok=True)
    output_dir = f"./performance_logs/{folder_name}"
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))
    
    console.print(f"[bold green]Starting background monitoring to {output_dir}[/bold green]")
    console.print("Logs will be saved to:")
    console.print(f"  [cyan]GPU:[/cyan] {os.path.join(output_dir, 'gpu_performance.csv')}")
    console.print(f"  [cyan]CPU:[/cyan] {os.path.join(output_dir, 'cpu_performance.csv')}")
    console.print(f"[yellow]Monitoring interval:[/yellow] {interval} seconds")
    console.print("[dim]Use 'ps aux | grep perfomance_load' to find the process[/dim]")
    console.print("[dim]Use 'kill <pid>' to stop monitoring[/dim]")
    monitor_performance(output_dir, interval)

@app.command()
def training(
    config_file: str = typer.Option(..., "--config", "-c", help="The path to the config file (required)"),
    experiment_name: str = typer.Option(..., "--experiment-name", "-e", help="The name of the experiment (required)"),
    training_data: str = typer.Option(..., "--train-data", "-t", help="The path to the train data (required)"),
    lr: float = typer.Option(0.001, "--lr", "-l", help="Learning rate (default: 0.001)"),
    batch_max_length: int = typer.Option(34, "--batch-max-length", "-b", help="Maximum batch length (default: 34)"),
    workers: int = typer.Option(16, "--workers", "-w", help="Number of data loading workers (default: 16)"),
    batch_size: int = typer.Option(32, "--batch-size", "-bs", help="Batch size (default: 32)"),
    num_iter: int = typer.Option(10000, "--num-iter", "-ni", help="Number of training iterations (default: 10000)"),
    valInterval: int = typer.Option(100, "--val-interval", "-vi", help="Validation interval (default: 100)"),
    FT: bool = typer.Option(True, "--FT", "-ft", help="Fine-tune the model"),
    imgH: int = typer.Option(64, "--img-height", "-ih", help="Image height (default: 64)"),
    imgW: int = typer.Option(600, "--img-width", "-iw", help="Image width (default: 600)"),
    sensitive: bool = typer.Option(True, "--sensitive", "-s", help="Use sensitive training (default: True)"),
    amp: bool = typer.Option(False, "--amp", "-a", help="Use automatic mixed precision training"),
):
    """
    Run training using the specified config file.

    example:
    python training_pipeline.py training --config config_files/th_custom_config.yaml --experiment-name th_filtered_2gpus --train-data all_data/<dataset> --lr 0.001 --batch-max-length 34 --workers 16 --batch-size 32 --num-iter 10000 --val-interval 100 --FT --img-height 64 --img-width 600 --sensitive --amp
    python training_pipeline.py training -c config_files/th_custom_config.yaml -e <experiment_name> -t all_data/<dataset> -l 0.001 -b 34 -w 16 -bs 32 -ni 10000 -vi 100 -ft -ih 64 -iw 600 -s -a

    example-full:
    python training_pipeline.py training --config config_files/th_custom_config.yaml --experiment-name my_experiment --train-data all_data/my_dataset --lr 0.001 --batch-max-length 34 --workers 16 --batch-size 32 --num-iter 10000 --val-interval 100 --FT --img-height 64 --img-width 600 --sensitive --amp
    """
    print("GPU list: ", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))

    cudnn.benchmark = True
    cudnn.deterministic = False

    def get_config(file_path, experiment_name, training_data):
        with open(file_path, 'r', encoding="utf8") as stream:
            opt = yaml.safe_load(stream)
        opt = AttrDict(opt)
        if opt.lang_char == 'None':
            characters = ''
            for data in opt['select_data'].split('-'):
                csv_path = os.path.join(training_data, data, 'labels.csv')
                df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
                all_char = ''.join(df['words'])
                characters += ''.join(set(all_char))
            characters = sorted(set(characters))
            opt.character= ''.join(characters)
        else:
            opt.character = opt.number + opt.symbol + opt.lang_char
        os.makedirs(f'./saved_models/{experiment_name}', exist_ok=True)
        return opt

    opt = get_config(config_file, experiment_name, training_data)

    opt.experiment_name = experiment_name
    opt.train_data = training_data
    opt.valid_data = os.path.join(training_data, "val")
    opt.lr = lr
    opt.batch_max_length = batch_max_length
    opt.imgH = imgH
    opt.imgW = imgW
    opt.workers = workers
    opt.batch_size = batch_size
    opt.num_iter = num_iter
    opt.valInterval = valInterval
    opt.FT = FT
    opt.sensitive = sensitive
    console.print(opt)

    # AttrDict has no to_dict method; convert directly to a plain dict for YAML dumping
    config_path = f'./saved_models/{experiment_name}/config.yaml'
    with open(config_path, 'w', encoding="utf8") as stream:
        yaml.safe_dump(dict(opt), stream, allow_unicode=True, sort_keys=False)
    console.print(f"[green]Saved config to[/green] {config_path}")
    train(opt, amp=amp)

if __name__ == "__main__":
    app()