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
parser.add_argument('--data_repo', type=str, default="pathtracker")
parser.add_argument('--channels_color', type=int, default=3)
parser.add_argument('--im_size', type=int, default=32)
parser.add_argument('--init_phases', type=str, default='ideal', choices=['ideal', 'ideal2','ideal3', 'random', 'learnable', 'last', 'cae', 'tag'])

parser.add_argument('--data_dir', type=str, default="feature_space_change_pathtracker-main")

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

# ========================= Kuramoto related ==========================
parser.add_argument('--distractor_masks', type=eval, default=True, choices=[True, False])
parser.add_argument('--kuramoto_channels', type=int, default=8)
parser.add_argument('--epsilon', type=float, default=0.202)
parser.add_argument('--epsilon_l2', type=float, default=0)
parser.add_argument('--lr_kuramoto', type=float, default=0.006)
parser.add_argument('--lr_kuramoto_l2', type=float, default=0.006)
parser.add_argument('--lr_kuramoto_l3', type=float, default=0.006)
parser.add_argument('--lr_kuramoto_l4', type=float, default=0.006)
parser.add_argument('--mean_r', type=float, default=0)
parser.add_argument('--std_g', type=float, default=3)
parser.add_argument('--std_r', type=float, default=3)
parser.add_argument('--coef', type=float, default=1)
parser.add_argument('--k', type=int, default=13)
parser.add_argument('--k_l2', type=int, default=11)
parser.add_argument('--k_l3', type=int, default=11)
parser.add_argument('--loss_coef1', type=float, default=1)
parser.add_argument('--loss_coef2', type=float, default=1)
parser.add_argument('--timesteps', type=int, default=15)
parser.add_argument('--from_input', type=eval, default=False, choices=[True, False])

parser.add_argument('--coef_green', type=float, default=1.5)
parser.add_argument('--coef_red', type=float, default=5.)
parser.add_argument('--coef_red_green', type=float, default=80.)
parser.add_argument('--neg_coef', type=float, default=1.5)
parser.add_argument('--learnable', type=eval, default=False, choices=[True, False])

parser.add_argument('--track_coef_green', type=float, default=10)
parser.add_argument('--track_coef_red', type=float, default=1)
parser.add_argument('--track_coef_blue', type=float, default=2)
parser.add_argument('--track_coef_red_green', type=float, default=-1)
parser.add_argument('--track_coef_green_blue', type=float, default=-1)
parser.add_argument('--track_std_g', type=float, default=0.8)
parser.add_argument('--track_std_r', type=float, default=0.7)
parser.add_argument('--track_k', type=int, default=3)
parser.add_argument('--track_lr_kuramoto', type=float, default=0.08)
parser.add_argument('--track_timesteps', type=int, default=1)
parser.add_argument('--track_init_end', type=eval, default=False, choices=[True, False])



