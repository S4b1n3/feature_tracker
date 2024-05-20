#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:04:57 2019

"""

import os
import time
import torch
torch.cuda.empty_cache()
from torchvision.transforms import Compose as transcompose
import torch.nn.parallel
from torch import nn
import torch.optim
import numpy as np
import wandb
# from utils.dataset import DataSetSeg
from utils.TFRDataset import tfr_data_loader

from utils.transforms import GroupScale, Augmentation, Stack, ToTorchFormatTensor
from utils.misc_functions import AverageMeter, FocalLoss, acc_scores, save_checkpoint
from statistics import mean
from utils.opts import parser
from utils import presets
from utils import engine
from utils.earlystopping import EarlyStopping
import matplotlib
# import imageio
from torch._six import inf
# from torchvideotransforms import video_transforms, volume_transforms


torch.backends.cudnn.benchmark = False
# torch.autograd.set_detect_anomaly(True)


global best_prec1
best_prec1 = 0
args = parser.parse_args()
# video_transform_list = [video_transforms.RandomHorizontalFlip(0.5), video_transforms.RandomVerticalFlip(0.5)]  # , volume_transforms.ClipToTensor(div_255=False)]
# transforms = video_transforms.Compose(video_transform_list)
use_augmentations = False
disentangle_channels = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login()
run = wandb.init(project="tracking", name=args.name, config=args)
args.name = args.name+"_"+run.id


def debug_plot(img):
    from matplotlib import pyplot as plt
    import tkinter
    import matplotlib
    matplotlib.use('TkAgg')
    pl = img.cpu().numpy().transpose(0, 2, 3, 4, 1)
    plt.subplot(131);plt.imshow(pl[0, 0]);plt.subplot(132);plt.imshow(pl[0, 5]);plt.subplot(133);plt.imshow(pl[0, 10]);plt.show()


def validate(val_loader, model, criterion, device, logiters=None, test=False):
    batch_timev = AverageMeter()
    lossesv = AverageMeter()
    top1v = AverageMeter()
    precisionv = AverageMeter()
    recallv = AverageMeter()
    f1scorev = AverageMeter()


    end = time.time()
    with torch.no_grad():
        for i, (imgs, target) in enumerate(val_loader):
            independent_frames = False
            imgs, masks, target = engine.prepare_data(imgs=imgs, target=target, args=args, device=device, disentangle_channels=disentangle_channels, independent_images=independent_frames)  # noqa

            # debug_plot(imgs)
            output, jv_penalty = engine.model_step(model, imgs, masks, model_name=args.model, test=True, cae=cae)
            if 'CVInT' in args.model:
                output, loss_synch, prediction_loss = output
                loss_synch_scaled = args.loss_coef1 * loss_synch.mean()
                prediction_loss = args.loss_coef2 * prediction_loss
            else:
                loss_synch_scaled = 0
                prediction_loss = 0
            loss = criterion(output, target.float().reshape(-1, 1)) + loss_synch_scaled + prediction_loss
            prec1, preci, rec, f1s = acc_scores(target, output.data)

            lossesv.update(loss.data.item(), 1)
            top1v.update(prec1.item(), 1)
            precisionv.update(preci.item(), 1)
            recallv.update(rec.item(), 1)
            f1scorev.update(f1s.item(), 1)

            batch_timev.update(time.time() - end)
            end = time.time()

            # if (i % args.print_freq == 0 or (i == len(val_loader) - 1)) and logiters is None:
            if (i % args.print_freq == 0 or (i == len_val_loader - 1)) and logiters is None:
                if test:
                    name = 'Test'
                else:
                    name = 'Val'
                print_string = '{name}: [{0}/{1}]\t Time: {batch_time.avg:.3f}\t Loss: {loss.val:.8f} ({loss.avg: .8f})\t'\
                               'Bal_acc: {balacc:.8f} preci: {preci.val:.5f} ({preci.avg:.5f}) rec: {rec.val:.5f}'\
                               '({rec.avg:.5f}) f1: {f1s.val:.5f} ({f1s.avg:.5f})'\
                               .format(i * args.batch_size, len_val_loader, batch_time=batch_timev, loss=lossesv, balacc=top1v.avg,
                                       preci=precisionv, rec=recallv, f1s=f1scorev, name=name)
                if test:
                    wandb.log({"Test Loss": lossesv.avg, "Test Bal Acc": top1v.avg, "Test Precision": precisionv.avg, "Test Recall": recallv.avg, "Test F1 Score": f1scorev.avg})
                else:
                    wandb.log({"val_loss": lossesv.avg, "val_acc": top1v.avg, "val_prec": precisionv.avg, "var_rec": recallv.avg, "val_f1": f1scorev.avg})
                    if 'CVInT' in args.model:
                        wandb.log({'val_synch_loss': loss_synch.item(), 'val_scaled_loss': loss_synch_scaled, 'val_prediction_loss': prediction_loss})
                print(print_string)
                with open(results_folder + args.name + '.txt', 'a+') as log_file:
                    log_file.write(print_string + '\n')


            elif logiters is not None:
                if i > logiters:
                    break
    model.train()
    return top1v.avg, precisionv.avg, recallv.avg, f1scorev.avg, lossesv.avg





def save_npz(epoch, log_dict, results_folder, savename='train'):

    with open(results_folder + savename + '.npz', 'wb') as f:
        np.savez(f, **log_dict)


if __name__ == '__main__':
    
    assert args.dist is not None, "You must pass a PT distance."
    assert args.speed is not None, "You must pass a PT speed."
    assert args.length is not None, "You must pass a PT length."
    stem = "{}_{}_{}".format(args.length, args.speed, args.dist)
    wandb.config.update(args, allow_val_change=True)
    pf_root, timesteps, len_train_loader, len_val_loader, len_test_loader = engine.dataset_selector(dist=args.dist, speed=args.speed, length=args.length, data_repo=args.data_repo, optical_flow=args.optical_flow)  # 14, 1, 64
    print(pf_root)
    print("Loading training dataset")
    train_loader = tfr_data_loader(data_dir=pf_root+'train-*', batch_size=args.batch_size, drop_remainder=True, timesteps=args.length, channels_color=args.channels_color, im_size=args.im_size)  # , optical_flow=args.optical_flow)
    # import pdb; pdb.set_trace()

    print("Loading validation dataset")
    val_loader = tfr_data_loader(data_dir=pf_root+'val-*', batch_size=args.batch_size, drop_remainder=True, timesteps=args.length, channels_color=args.channels_color, im_size=args.im_size)  # , optical_flow=args.optical_flow)

    print("Loading validation dataset")
    test_loader = tfr_data_loader(data_dir=pf_root + 'test-*', batch_size=args.batch_size, drop_remainder=True,
                                 timesteps=args.length, channels_color=args.channels_color, im_size=args.im_size)  # , optical_flow=args.optical_flow)

    if args.optical_flow:
        stem = "_{}".format(stem, "flow")
    # results_folder = os.path.join('results', stem, '{0}'.format(args.name))
    results_folder = os.path.join("./results", stem, '{0}/'.format(args.name))
    ES = EarlyStopping(patience=500, results_folder=results_folder)
    os.makedirs(results_folder, exist_ok=True)
    exp_logging = args.log
    jacobian_penalty = args.penalty

    model = engine.model_selector(args=args, timesteps=timesteps, device=device, dimensions=args.dimensions)

    cae = None
    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
    if args.parallel is True:
        model = torch.nn.DataParallel(model).to(device)
        print("Loading parallel finished on GPU count:", torch.cuda.device_count())
    else:
        model = model.to(device)
        print("Loading finished")

    # noqa Save timesteps/kernel_size/dimensions/learning rate/epochs/exp_name/algo/penalty to a dict for reloading in the future
    param_names_shapes = {k: v.shape for k, v in model.named_parameters()}
    hp_dict = {
        "penalty": jacobian_penalty,
        "start_epoch": args.start_epoch,
        "epochs": args.epochs,
        "lr": args.learning_rate,
        "loaded_ckpt": args.ckpt,
        "results_dir": results_folder,
        "exp_name": args.name,
        "algo": args.algo,
        "dimensions": args.dimensions,
        "fb_kernel_size": args.fb_kernel_size,
        "param_names_shapes": param_names_shapes,
        "timesteps": timesteps
    }
    np.savez(os.path.join(results_folder, "hp_dict"), **hp_dict)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    print("Including parameters {}".format([k for k, v in model.named_parameters()]))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.7)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-2)
    lr_init = args.learning_rate

    val_log_dict = {'loss': [], 'balacc': [], 'precision': [], 'recall': [], 'f1score': []}
    train_log_dict = {'loss': [], 'balacc': [], 'precision': [], 'recall': [], 'f1score': [], 'jvpen': [], 'scaled_loss': []}

    if args.ckpt is not None and args.init_phases != 'cae':
        model = engine.load_ckpt(model, args.ckpt)
    scale = torch.Tensor([1.0]).to(device)
    for epoch in range(args.start_epoch, args.epochs):
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        precision = AverageMeter()
        recall = AverageMeter()
        f1score = AverageMeter()

        time_since_last = time.time()
        model.train()
        end = time.perf_counter()

        model.eval()
        accv, precv, recv, f1sv, losv = validate(val_loader, model, criterion, device, logiters=3)
        model.train()
        for idx, (imgs, target) in enumerate(train_loader):
            #with torch.autocast(device_type='cuda', dtype=torch.float16):
            data_time.update(time.perf_counter() - end)
            if args.init_phases == 'cae':
                independent_frames = True
            else:
                independent_frames = False
            imgs, masks, target = engine.prepare_data(imgs=imgs, target=target, args=args, device=device, disentangle_channels=disentangle_channels, independent_images=independent_frames)  # noqa


            # Run training
            output, jv_penalty = engine.model_step(model, imgs, masks, model_name=args.model, cae=cae)
            if 'CVInT' in args.model:
                output, loss_synch, prediction_loss = output
                loss_synch_scaled = args.loss_coef1 * loss_synch.mean()
                prediction_loss = args.loss_coef2 * prediction_loss
            else:
                loss_synch_scaled = 0
                prediction_loss = 0
            loss = criterion(output, target.float().reshape(-1, 1)) + loss_synch_scaled + prediction_loss
            losses.update(loss.data.item(), 1)
            jv_penalty = jv_penalty.mean()
            train_log_dict['jvpen'].append(jv_penalty.item())

            if jacobian_penalty:
                loss = loss + jv_penalty * 1e1

            prec1, preci, rec, f1s = acc_scores(target[:], output.data[:])
            top1.update(prec1.item(), 1)
            precision.update(preci.item(), 1)
            recall.update(rec.item(), 1)
            f1score.update(f1s.item(), 1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_time.update(time.perf_counter() - end)
            
            end = time.perf_counter()
            if idx % (args.print_freq) == 0:
                time_now = time.time()
                print_string = 'Epoch: [{0}][{1}/{2}]  lr: {lr:g}  Time: {batch_time.val:.3f} (itavg:{timeiteravg:.3f}) '\
                               '({batch_time.avg:.3f})  Data: {data_time.val:.3f} ({data_time.avg:.3f}) ' \
                               'Loss: {loss.val:.8f} ({lossprint:.8f}) ({loss.avg:.8f})  bal_acc: {top1.val:.5f} '\
                               '({top1.avg:.5f}) preci: {preci.val:.5f} ({preci.avg:.5f}) rec: {rec.val:.5f} '\
                               '({rec.avg:.5f})  f1: {f1s.val:.5f} ({f1s.avg:.5f}) jvpen: {jpena:.12f} {timeprint:.3f} losscale:{losscale:.5f}'\
                               .format(epoch, idx, len_train_loader, batch_time=batch_time, data_time=data_time, loss=losses,
                                       lossprint=mean(losses.history[-args.print_freq:]), lr=optimizer.param_groups[0]['lr'],
                                       top1=top1, timeiteravg=mean(batch_time.history[-args.print_freq:]),
                                       timeprint=time_now - time_since_last, preci=precision, rec=recall,
                                       f1s=f1score, jpena=jv_penalty.item(), losscale=scale.item())
                print(print_string)
                time_since_last = time_now
                with open(results_folder + args.name + '.txt', 'a+') as log_file:
                    log_file.write(print_string + '\n')
                wandb.log({'train_loss': losses.val, 'train_acc': top1.val, 'train_prec': precision.val, 'train_rec': recall.val, 'train_f1': f1score.val, 'train _jv_penalty': jv_penalty.item(), 'train_lr': optimizer.param_groups[0]['lr']})
                if 'CVInT' in args.model:
                    wandb.log({'train_synch_loss': loss_synch.item(), 'train_scaled_loss': loss_synch_scaled, 'train_prediction_loss': prediction_loss})

        # lr_scheduler.step()
        print(epoch)
        train_log_dict['loss'].extend(losses.history)
        train_log_dict['balacc'].extend(top1.history)
        train_log_dict['precision'].extend(precision.history)
        train_log_dict['recall'].extend(recall.history)
        train_log_dict['f1score'].extend(f1score.history)
        save_npz(epoch, train_log_dict, results_folder, 'train')
        save_npz(epoch, val_log_dict, results_folder, 'val')

        if (epoch + 1) % 1 == 0 or epoch == args.epochs - 1:
            model.eval()
            accv, precv, recv, f1sv, losv = validate(val_loader, model, criterion, device, logiters=3)
            model.train()
            print_string = 'val f {} val loss {}'.format(f1sv, losv)
            print(print_string)
            val_log_dict['loss'].append(losv)
            val_log_dict['balacc'].append(accv)
            val_log_dict['precision'].append(precv)
            val_log_dict['recall'].append(recv)
            val_log_dict['f1score'].append(f1sv)
            wandb.log(
                {"val_loss": losv, "val_acc": accv, "val_prec": precv, "var_rec": recv,
                 "val_f1": f1sv})

            with open(results_folder + args.name + '.txt', 'a+') as log_file:
                log_file.write(print_string + '\n')
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc': accv,
                'best_loss': losv}, True, results_folder)

            # # save a model checkpoint at the end of each epoch to W&B
            # model_artifact = wandb.Artifact(
            #     name=args.name,
            #     type="model")
            #
            # # Add your model weights file to the artifact
            # model_artifact.add_file("model.pt")
            #
            # # log the Artifact to W&B
            # wandb.log_artifact(
            #     model_artifact,
            #     aliases=[f"epoch - {epoch + 1}", f"val_accuracy - {accv}"])


        #     ES(losv, model, epoch)
        # if ES.early_stop:
        #     print("Early stopping triggered. Quitting.")
        #     os._exit(1)

    model.eval()
    accv, precv, recv, f1sv, losv = validate(test_loader, model, criterion, device, logiters=3, test=True)
    wandb.log(
        {"test_loss": losv, "test_acc": accv, "test_prec": precv, "test_rec": recv,
         "test_f1": f1sv})

    print_string = 'test f {} test loss {}'.format(f1sv, losv)
    print(print_string)
    val_log_dict['loss'].append(losv)
    val_log_dict['balacc'].append(accv)
    val_log_dict['precision'].append(precv)
    val_log_dict['recall'].append(recv)
    val_log_dict['f1score'].append(f1sv)
    with open(results_folder + args.name + '.txt', 'a+') as log_file:
        log_file.write(print_string + '\n')


