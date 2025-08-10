import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import csv  # เพิ่ม import csv
from contextlib import nullcontext

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
from rich.console import Console
from rich.table import Table
"""Training script with enhanced Rich console logging."""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
console = Console()

def count_parameters(model):
    table = Table(title="Trainable Parameters", show_lines=False)
    table.add_column("Module", overflow="fold")
    table.add_column("#Params", justify="right")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
        table.add_row(name, f"{param:,}")
    console.print(table)
    console.log(f"[bold green]Total Trainable Params:[/bold green] {total_params:,}")
    return total_params

def train(opt, show_number = 2, amp=False):
    """ dataset preparation """
    if not opt.data_filtering_off:
        console.log('[cyan]Filtering the images containing characters not in charset[/cyan]')
        console.log('[cyan]Filtering images whose label is longer than batch_max_length[/cyan]')

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    console.log('[bold]Initializing training datasets...[/bold]')
    dataset_init_start = time.time()
    train_dataset = Batch_Balanced_Dataset(opt)
    console.log(f"Training dataset ready in {time.time()-dataset_init_start:0.2f}s")

    log = open(f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a', encoding="utf8")
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=min(32, opt.batch_size),
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers), 
        prefetch_factor=512 if int(opt.workers) > 0 else None,
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    console.rule("Dataset Summary")
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    console.log('Model input parameters: ' + 
                f"H={opt.imgH} W={opt.imgW} fiducial={opt.num_fiducial} in_ch={opt.input_channel} out_ch={opt.output_channel} "
                f"hidden={opt.hidden_size} classes={opt.num_class} max_len={opt.batch_max_length} "
                f"T={opt.Transformation} FE={opt.FeatureExtraction} Seq={opt.SequenceModeling} Pred={opt.Prediction}")

    if opt.saved_model != '':
        pretrained_dict = torch.load(opt.saved_model)
        if opt.new_prediction:
            model.Prediction = nn.Linear(model.SequenceModeling_output, len(pretrained_dict['module.Prediction.weight']))

        model = torch.nn.DataParallel(model).to(device)
        console.log(f'[yellow]Loading pretrained model[/yellow] from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.load_state_dict(pretrained_dict)
        if opt.new_prediction:
            model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, opt.num_class)
            for name, param in model.module.Prediction.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            model = model.to(device)
    else:
        # weight initialization
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                console.log(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        model = torch.nn.DataParallel(model).to(device)
    
    model.train() 
    console.rule("Model Architecture")
    console.print(model)
    count_parameters(model)
    
    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # freeze some layers
    try:
        if opt.freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if opt.freeze_SequenceModeling:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except Exception:
        pass
    
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    console.log(f'Trainable params num: {sum(params_num):,}')
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.optim=='adam':
        #optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizer = optim.Adam(filtered_parameters)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    console.rule("Optimizer")
    console.print(optimizer)

    """ final options """
    # print(opt)
    opt_log = '------------ Options -------------\n'
    args = vars(opt)
    for k, v in args.items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '---------------------------------------\n'
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a', encoding="utf8") as opt_file:
        opt_file.write(opt_log)
    console.print(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            console.log(f'[yellow]Continue training[/yellow] from iteration {start_iter}')
        except Exception:
            pass

    # สร้างไฟล์ CSV และเขียน header
    csv_path = f'./saved_models/{opt.experiment_name}/training_log.csv'
    csv_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    
    # เขียน header ถ้าเป็นไฟล์ใหม่
    if not csv_exists or start_iter == 0:
        csv_writer.writerow(['iteration', 'epoch', 'train_loss', 'valid_loss', 'accuracy', 'norm_ED', 
                           'best_accuracy', 'best_norm_ED', 'elapsed_time', 'learning_rate'])
    
    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter

    scaler = GradScaler()
    t1= time.time()
        
    # Timing accumulators (per validation interval)
    interval_data_time = 0.0
    interval_forward_time = 0.0
    interval_backward_time = 0.0
    interval_opt_step_time = 0.0
    interval_iter_time = 0.0
    interval_start_time = time.time()
    last_log_iter = start_iter
    while True:
        # train part
        iter_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        # ---------------- Data Loading ----------------
        t_data_start = time.time()
        image_tensors, labels = train_dataset.get_batch()
        data_time = time.time() - t_data_start
        interval_data_time += data_time
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        # ---------------- Forward & Loss ----------------
        t_fwd_start = time.time()
        amp_ctx = autocast() if amp else nullcontext()
        with amp_ctx:
            if 'CTC' in opt.Prediction:
                preds = model(image, text).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
                preds_perm = preds.permute(1, 0, 2)
                torch.backends.cudnn.enabled = False
                cost = criterion(preds_perm, text.to(device), preds_size, length.to(device))
                torch.backends.cudnn.enabled = True
            else:
                preds = model(image, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        fwd_time = time.time() - t_fwd_start
        interval_forward_time += fwd_time

        # ---------------- Backward ----------------
        t_bwd_start = time.time()
        if amp:
            scaler.scale(cost).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        else:
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        bwd_time = time.time() - t_bwd_start
        interval_backward_time += bwd_time

        # ---------------- Optimizer Step ----------------
        t_opt_start = time.time()
        if amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        opt_time = time.time() - t_opt_start
        interval_opt_step_time += opt_time

        iter_time = time.time() - iter_start
        interval_iter_time += iter_time
        
        loss_avg.add(cost)

        # validation part
        if (i % opt.valInterval == 0) and (i!=0):
            interval_wall = time.time() - interval_start_time
            console.log(f"[magenta]Interval {last_log_iter}->{i}[/magenta] time: {interval_wall:0.2f}s | "
                        f"iter_avg={interval_iter_time/(i-last_log_iter):0.3f}s (data {interval_data_time/(i-last_log_iter):0.3f} | "
                        f"fwd {interval_forward_time/(i-last_log_iter):0.3f} | bwd {interval_backward_time/(i-last_log_iter):0.3f} | opt {interval_opt_step_time/(i-last_log_iter):0.3f}) | "
                        f"throughput={( (i-last_log_iter)*batch_size )/interval_wall:0.1f} samples/s")
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                console.log(f"GPU Memory Allocated: {mem_alloc:0.1f}MB | Reserved: {mem_reserved:0.1f}MB")

            # reset interval timers
            interval_data_time = interval_forward_time = interval_backward_time = interval_opt_step_time = interval_iter_time = 0.0
            interval_start_time = time.time()
            last_log_iter = i
            console.log('training time: ' + f"{time.time()-t1:0.2f}s")
            t1=time.time()
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a', encoding="utf8") as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                    infer_time, length_of_data = validation(model, criterion, valid_loader, converter, opt, device)
                model.train()

                # training loss and validation loss
                loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                
                # คำนวณ epoch (สมมติว่า 1 epoch = จำนวน iteration ที่ใช้ไปทั้งหมดของ dataset)
                epoch = i // len(train_dataset) if hasattr(train_dataset, '__len__') else i // 1000
                
                # ดึง learning rate ปัจจุบัน
                current_lr = optimizer.param_groups[0]['lr']
                
                # บันทึกข้อมูลลง CSV
                csv_writer.writerow([
                    i,  # iteration
                    epoch,  # epoch
                    f'{loss_avg.val():.5f}',  # train_loss
                    f'{valid_loss:.5f}',  # valid_loss
                    f'{current_accuracy:.3f}',  # accuracy
                    f'{current_norm_ED:.4f}',  # norm_ED
                    f'{best_accuracy:.3f}',  # best_accuracy
                    f'{best_norm_ED:.4f}',  # best_norm_ED
                    f'{elapsed_time:.2f}',  # elapsed_time
                    f'{current_lr:.6f}'  # learning_rate
                ])
                csv_file.flush()  # ทำให้แน่ใจว่าข้อมูลถูกเขียนลงไฟล์ทันที
                
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                console.log(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                
                #show_number = min(show_number, len(labels))
                
                start = random.randint(0,len(labels) - show_number )    
                for gt, pred, confidence in zip(labels[start:start+show_number], preds[start:start+show_number], confidence_score[start:start+show_number]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                console.print(predicted_result_log)
                log.write(predicted_result_log + '\n')
                console.log('validation time: ' + f"{time.time()-t1:0.2f}s")
                t1=time.time()
        # save model per 1e+4 iter.
        if (i + 1) % 1e+4 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.experiment_name}/iter_{i+1}.pth')

        if i == opt.num_iter:
            console.log('[bold green]End training[/bold green]')
            csv_file.close()  # ปิดไฟล์ CSV ก่อนจบโปรแกรม
            sys.exit()
        i += 1