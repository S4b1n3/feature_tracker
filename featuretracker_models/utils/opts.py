import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of hGRU")

# parser.add_argument('train_list', type=str)
# parser.add_argument('val_list', type=str)

parser.add_argument('--name', type=str, default="hgru")
parser.add_argument('--name_pretrained', type=str, default="hgru")

parser.add_argument('--model', type=str, default="hgru")
parser.add_argument('--algo', type=str, default="bptt")
parser.add_argument('--penalty', default=False, action='store_true')
parser.add_argument('--pretrained', type=eval, default=False, choices=[True, False])
parser.add_argument('--optical_flow', default=False, action='store_true')

parser.add_argument('--epoch_test', type=int, default=None)

parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--dist', type=int)
parser.add_argument('--speed', type=int)
parser.add_argument('--length', type=int)
parser.add_argument('--data_repo', type=str, default="idshapes_idcolors")
parser.add_argument('--channels_color', type=int, default=3)
parser.add_argument('--im_size', type=int, default=32)

parser.add_argument('--data_dir', type=str, default="feature_tracker")
parser.add_argument('--loss_coef1', type=float, default=1)

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')


parser.add_argument('-d', '--dimensions', default=32, type=int)
parser.add_argument('-k', '--fb_kernel_size', default=7, type=int)
parser.add_argument('--nb_rec_units', default=1, type=int)

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('-parallel', '--parallel', type=eval, default=True, choices=[True, False],
                    help='Wanna parallelize the training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--log', type=eval, default=True, choices=[True, False])

parser.add_argument('--val-freq', '-vf', default=2000, type=int,
                    metavar='N', help='Validation frequency')


