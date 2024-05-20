import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from models import CVInT
from models import InT
import imageio
from torchvision.models import video
from models.slowfast_utils import slowfast, slowfast_nl
from models import transformers
from models import kys
from pytorchvideo.layers.positional_encoding import SpatioTemporalClsPositionalEncoding
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.stem import create_conv_patch_embed
try:
    from models import resnet_TSM as rntsm
except:
    print("Failed to import spatial sampler.")
try:
    from tqdm import tqdm
except:
    print("Failed to import tqdm.")
from utils.opts import parser
args = parser.parse_args()


TORCHVISION = ['r3d', 'mc3', 'r2plus1', 'nostride_r3d', 'nostride_r3d_pos', 'mvit']
SLOWFAST = ['slowfast', 'slowfast_nl']
REC = ['CVInT']
ALL_DATASETS = [
    {"dist": 10, "speed": 1, "length": 32},
    # {"dist": 3, "speed": 1, "length": 32},
    # {"dist": 10, "speed": 1, "length": 32},
    # {"dist": 16, "speed": 1, "length": 32},
    # {"dist": 20, "speed": 1, "length": 32},
    # {"dist": 26, "speed": 1, "length": 32},
]

def model_step(model, imgs, masks, model_name, test=False, cae=None):
    """Pass imgs through the model."""
    if model_name in TORCHVISION:
        output = model.forward(imgs)
        jv_penalty = torch.tensor([1]).float().cuda() 
    elif model_name in SLOWFAST:
        # frames = F.interpolate(imgs, 224, mode='trilinear', align_corners=True)  # F.interpolate(imgs, 224)
        frames = imgs
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        ALPHA = 4 
        slow_pathway = torch.index_select(
            frames,
            2,  # 1
            torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // ALPHA, device=imgs.device
            ).long(),
            )  # Note that this usually operates per-exemplar. Here we do per batch.
        frame_list = [slow_pathway, fast_pathway]
        output = model.forward(frame_list)
        jv_penalty = torch.tensor([1]).float().cuda()
    elif model_name in REC:
        if test:
            output, jv_penalty = model.forward(imgs, masks, testmode=True)
            return output, jv_penalty
        else:
            output, jv_penalty = model.forward(imgs, masks)
    else:
        if test:
            output, states, gates = model.forward(imgs, testmode=True)
            return output, None #states, gates
        else:
            output, jv_penalty = model.forward(imgs)
    if test:
        return output, None #, None
    else:
        return output, jv_penalty


def model_selector(args, timesteps, device, fb_kernel_size=7, dimensions=32):
    """Select a model."""
    if args.model == 'CVInT':
        print("Init model CVInT ", args.algo, 'penalty: ', args.penalty)
        model = CVInT.CVInT(
            dimensions=dimensions,
            in_channels=args.channels_color,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            grad_method='bptt')
    elif args.model == 'InT':
        print("Init model InT ", args.algo, 'penalty: ', args.penalty)
        model = InT.InT(
            dimensions=dimensions,
            in_channels=args.channels_color,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            grad_method='bptt')
    elif args.model == 'timesformer':
        model = transformers.TransformerModel(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            grad_method='bptt')
    elif args.model == 'slowfast':
        model = slowfast()
    elif args.model == 'slowfast_nl':
        model = slowfast_nl()
    elif args.model == 'r3d':
        model = video.r3d_18(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        if args.channels_color != 3:
            model.stem = nn.Sequential(nn.Conv3d(args.channels_color, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True))
    elif args.model == 'swin3d':
        model = video.swin3d_t(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif args.model == 's3d':
        model = video.s3d(pretrained=args.pretrained, num_classes=1)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, 1)
    elif args.model == 'rntsm':
        model = rntsm.resnet50(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=1)
    elif args.model == 'mc3':
        model = video.mc3_18(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif args.model == 'r2plus1':
        model = video.r2plus1d_18(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif args.model == 'mvit' and args.pretrained:
        cls_positional_encoding = SpatioTemporalClsPositionalEncoding(
            embed_dim=96,
            patch_embed_shape=[args.length//2, args.im_size//4, args.im_size//4],
            sep_pos_embed=True,
            has_cls=True,
        )
        head = create_vit_basic_head(
            in_features=768,
            out_features=1,
        )
        model = torch.hub.load("facebookresearch/pytorchvideo", model="mvit_base_16x4", pretrained=args.pretrained)
        model.cls_positional_encoding = cls_positional_encoding
        model.head = head
        if args.channels_color != 3:
            patch_embed = create_conv_patch_embed(
                in_channels=args.channels_color,
                out_channels=96,
                conv_kernel_size=[3, 7, 7],
                conv_stride=[2, 4, 4],
                conv_padding=[1, 3, 3],
            )
            model.patch_embed = patch_embed

    elif args.model == 'mvit':
        model = torch.hub.load("facebookresearch/pytorchvideo", model="mvit_base_16x4", pretrained=args.pretrained,
                               spatial_size=args.im_size, temporal_size=args.length, head_num_classes=1, input_channels=args.channels_color)
    else:
        raise NotImplementedError("Model not found.")
    return model


def prepare_data(imgs, target, args, device, disentangle_channels, use_augmentations=False, independent_images=False):
    """Prepare the data for training or eval."""
    imgs_masks = imgs.numpy()
    imgs = imgs_masks[:,0].transpose(0,4,1,2,3)
    phase_masks = imgs_masks[:, 1].transpose(0, 4, 1, 2, 3)
    target = torch.from_numpy(np.vectorize(ord)(target.numpy()))
    target = target.to(device, dtype=torch.float)
    imgs = imgs / 255.  # Normalize to [0, 1]
    phase_masks = phase_masks / 255.

    if disentangle_channels:
        mask = imgs.sum(1).round()
        proc_imgs = np.zeros_like(imgs)
        proc_imgs[:, 1] = (mask == 1).astype(imgs.dtype)
        proc_imgs[:, 2] = (mask == 2).astype(imgs.dtype)
        thing_layer = (mask == 3).astype(imgs.dtype)
        proc_imgs[:, 0] = thing_layer
    else:
        proc_imgs = imgs
    if use_augmentations:
        imgs = transforms(proc_imgs)
        imgs = np.stack(imgs, 0)
    else:
        imgs = proc_imgs
    imgs = torch.from_numpy(proc_imgs)
    phase_masks = torch.from_numpy(phase_masks)
    if independent_images:
        imgs = imgs.permute(0, 2, 1, 3, 4)
        imgs = imgs.reshape(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
        phase_masks = phase_masks.permute(0, 2, 1, 3, 4)
        phase_masks = phase_masks.reshape(-1, phase_masks.shape[2], phase_masks.shape[3], phase_masks.shape[4])

    imgs = imgs.to(device, dtype=torch.float)
    phase_masks = phase_masks.to(device, dtype=torch.float)
    if args.pretrained:
        mu = torch.tensor([0.43216, 0.394666, 0.37645], device=device)[None, :, None, None, None]
        stddev = torch.tensor([0.22803, 0.22145, 0.216989], device=device)[None, :, None, None, None]
        imgs = (imgs - mu) / stddev

    if "_cc" in args.model and args.model != "nostride_video_cc_small":
        img_shape = imgs.shape
        hh, ww = torch.meshgrid(torch.arange(1, img_shape[3] + 1, device=imgs.device, dtype=imgs.dtype), torch.arange(1, img_shape[4] + 1, device=imgs.device, dtype=imgs.dtype))
        hh = hh[None, None, None].repeat(img_shape[0], 1, img_shape[2], 1, 1)
        ww = ww[None, None, None].repeat(img_shape[0], 1, img_shape[2], 1, 1)
        imgs = torch.cat([imgs, hh, ww], 1)
    return imgs, phase_masks, target


def load_ckpt(model, model_path):
    # from glob import glob
    # model_path = glob(model_path)
    print(model_path)
    checkpoint = torch.load(model_path[0])
    # Check if "module" is the first part of the key
    # import pdb; pdb.set_trace()
    # check = checkpoint['state_dict'].keys()[0]

    # sd = checkpoint['state_dict']
    # if "module" in checkpoint and not args.parallel:
    #     new_sd = {}
    #     for k, v in sd.items():
    #         new_sd[k.replace("module.", "")] = v
    #     sd = new_sd
    model.load_state_dict(checkpoint['state_dict'], strict=False) #['state_dict']
    return model


def plot_results(states, imgs, target, output, timesteps, gates=None, prep_gifs=False, results_folder=None, show_fig=False):
    states = states.detach().cpu().numpy()
    gates = gates.detach().cpu().numpy()
    cols = (timesteps / 8) + 1
    rng = np.arange(0, timesteps, 8)
    rng = np.concatenate((np.arange(0,timesteps,8), [timesteps-1]))
    img = imgs.cpu().numpy()
    # from matplotlib import pyplot as plt
    sel = target.float().reshape(-1, 1) == (output > 0).float()
    sel = sel.cpu().numpy()
    sel = np.where(sel)[0]
    sel = sel[0]
    fig = plt.figure()
    for idx, i in enumerate(rng):
        plt.subplot(3, cols, idx + 1)
        plt.axis("off")
        plt.imshow(img[sel, :, i].transpose(1, 2, 0))
        plt.title("Img")
        plt.subplot(3, cols, idx + 1 + cols)
        plt.axis("off")
        plt.imshow((gates[sel, i].squeeze() ** 2).mean(0))
        plt.title("Attn")
        plt.subplot(3, cols, idx + 1 + cols + (cols - 1))
        plt.title("Activity")
        plt.axis("off")
        plt.imshow(np.abs(states[sel, i].squeeze()))
    plt.suptitle("Batch acc: {}, Prediction: {}, Label: {}".format((target.reshape(-1).float() == (output.reshape(-1) > 0).float()).float().mean(), output[sel].cpu(), target[sel]))
    if results_folder is not None:
        plt.savefig(os.path.join(results_folder, "random_selection.pdf"))
    if show_fig:
        plt.show()
    plt.close(fig)

    if prep_gifs:
        assert isinstance(prep_gifs, int), "prep_gifs is an integer that says how many gifs to prepare"
        assert results_folder is not None, "if prepping gifs, also pass a results folder."
        for g in tqdm(range(prep_gifs), total=prep_gifs, desc="Making gifs"):
            gif_dir = os.path.join(results_folder, "gif_{}".format(g))
            os.makedirs(gif_dir, exist_ok=True)
            filenames = []
            min_gate, max_gate = None, None  # (gates[g] ** 2).reshape(img.shape[2], -1).min() * 0.75, (gates[g] ** 2).reshape(img.shape[2], -1).max() * 0.75
            min_act, max_act = None, None  # (states[g] ** 2).reshape(img.shape[2], -1).min() * 0.75, (states[g] ** 2).reshape(img.shape[2], -1).max() * 0.75
            for idx, i in tqdm(enumerate(range(img.shape[2])), total=img.shape[2], desc="Writing gif images"):  # Loop over all timesteps
                fig = plt.figure(dpi=100)
                plt.subplot(1, 3, 1)
                plt.axis("off")
                plt.imshow(img[g, :, i].transpose(1, 2, 0))
                plt.title("Img")
                plt.subplot(1, 3, 2)
                plt.axis("off")
                plt.imshow((gates[g, i].squeeze() ** 2).mean(0), vmin=min_gate, vmax=max_gate)
                plt.title("Attn")
                plt.subplot(1, 3, 3)
                plt.title("Activity")
                plt.axis("off")
                # plt.imshow(np.abs(states[g, i].squeeze()), vmin=min_act, vmax=max_act)
                plt.imshow(states[g, i].squeeze() ** 2, vmin=min_act, vmax=max_act)
                plt.suptitle("Prediction: {}, Label: {}".format(output[g].cpu() > 0., target[g].cpu() == 1.))
                out_path = os.path.join(gif_dir, "{}.png".format(idx))
                plt.savefig(out_path)
                plt.close(fig)
                filenames.append(out_path)
            # Now produce gif
            gif_path = os.path.join(gif_dir, "{}.gif".format(g))
            with imageio.get_writer(gif_path, mode='I') as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                    os.remove(filename)


LOCAL = "../../"+args.data_dir

def get_datasets():
    return ALL_DATASETS

def dataset_selector(dist, speed, length, data_repo, optical_flow=False, testmode=False):
    """Organize the datasets here."""
    stem = "tfrecords"
    if optical_flow:
        stem = "tfrecords_optic_flow"
    if testmode: #_64speed_half_cs
        # lp = os.path.join(LOCAL, "new_psycho_data/"+str(dist)+"dist_1h_1c_traj/tfrecords/" + data_repo + "/")

        lp = os.path.join(LOCAL, str(dist)+"dist_2h_1c/"+str(length)+"frames/" + data_repo + "/tfrecords/")
    else:
        lp = os.path.join(LOCAL, str(dist)+"dist_64speed_half_cs/"+str(length)+"frames/" + data_repo + "/tfrecords/")

    print(lp)
    if os.path.exists(lp):
        print("Loading data from local storage.")
        return lp, length, 100000, 10000, 10000
    else:
        return '../'+args.data_dir + '/'+str(dist)+'dist_2h_1c/'+str(length)+'frames/'+data_repo+'/tfrecords/', length, 100000, 10000, 10000



