import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import clip
from torchvision.transforms import Resize
wm = "Paint-by-Example"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


# utils functions
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


# dist utils
def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def expand_first_dim(x, num_repeats=5, unsqueeze=False):
    if unsqueeze:
        x = x.unsqueeze(0)
    x = x.repeat(num_repeats, *([1] * (x.dim() - 1)))
    return x


def dist_init():
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


# val dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.transforms as T
import pycocotools.mask as maskUtils
import random
from torchvision.transforms import InterpolationMode, Compose, Resize, ToTensor, Normalize, CenterCrop
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _tokenzie_text(text: str):
    """Tokenize text."""
    return clip.tokenize(text, context_length=77, truncate=True) # 1x77


def _find_tokens_to_replace(class_name, text):
    """Find the tokens to be replaced."""
    if f"{class_name.lower()}s" in text.lower():
        class_name = class_name + "s"
    elif f"{class_name.lower()}es" in text.lower():
        class_name = class_name + "es"
    tokenized_class_name = _tokenzie_text(class_name).squeeze(0)
    tokenized_text = _tokenzie_text(text).squeeze(0)
    tokens_to_replace = []
    valid_tokens = tokenized_class_name[
            (tokenized_class_name != 49406) & \
            (tokenized_class_name != 49407) & \
            (tokenized_class_name != 0)]
    len_valid_tokens = len(valid_tokens)
    for i in range(len(tokenized_text) - len_valid_tokens + 1):
        if torch.equal(tokenized_text[i:i+len_valid_tokens], valid_tokens):
            tokens_to_replace.extend(
                list(range(i, i+len_valid_tokens)))

    return tokenized_class_name, tokenized_text, tokens_to_replace


# def _transform(n_px):
#     return Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])

# if dropping ration > 2, some objects like human, guitar, etc. will be dropped
# that is not what we want ...
def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def denormalize(img):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(img.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(img.device)
    # print("Before ", img.min(), img.max())
    img = img * std[None, :, None, None] + mean[None, :, None, None]
    # print("After ", img.min(), img.max())
    return img


class SAValDataset2(torch.utils.data.Dataset):

    def __init__(
            self, 
            path, 
            split="train", 
            splits=(0.9, 0.05, 0.05),
            crop_res=256, 
            min_resize_res=256,
            max_resize_res=256,
            flip_prob=0.5, 
            max_size=-1,
            bg="none",
            **kwargs):
        # data_path = DATA_PATH_TABLE["openimages"]
        self.data_path = path
        self.split = split
        # self.task_indicator = task_indicator
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res

        # fix the path now
        # self.filtered_ids = torch.load("../filtered_ids.pt", map_location="cpu")
        self.meta_info = torch.load(os.path.join(self.data_path, "val_999.pt"), map_location="cpu")
        if max_size is not None and max_size > 0:
            self.meta_info = self.meta_info[:max_size]
        
        self.bg = bg
        
    def _process_img(self, img, box, mask):
        # resize the image to the size of mask
        mask = mask.unsqueeze(0).float()
        mask = TF.resize(mask, (img.shape[1], img.shape[2]))
        mask = (mask > 0.5).float()
        # raw mask
        raw_mask = mask.clone()
        # generate a random number ranging in [0, 1]
        p1 = random.random()
        if p1 < 0.5:
            mask = F.conv2d(mask, torch.ones(1, 1, 3, 3), padding=1)
        else:
            mask = F.conv2d(mask, torch.ones(1, 1, 5, 5), padding=2)
        mask = (mask > 0.5).float()

        p = random.random()
        # if 0.0 <= p < 0.25:
        assert self.bg in ["white", "black", "none"]
        if self.bg == "white":
            # multiply the mask with the image
            img = img * mask
            extended_mask = mask.repeat(3, 1, 1)
            img[extended_mask <= 0.0] = 1.0
        # elif 0.25 <= p < 0.5:
        elif self.bg == "black":
            img = img * mask
        else:
            pass
        # crop the image using the 4-d box range in [0, 1]
        bbox = [
            int((box[1]         ) * img.shape[1]), # top
            int((box[0]         ) * img.shape[2]), # left
            int((box[3] - box[1]) * img.shape[1]), # height
            int((box[2] - box[0]) * img.shape[2]), # width
        ]
        # random jitter the box in form of [top, left, height, width] with 2 pixels
        offset = torch.randint(0, 3, (4,))
        bbox = [bbox[0] - offset[0], bbox[1] - offset[1], bbox[2] + offset[2], bbox[3] + offset[3]]
        # keep the range
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(img.shape[1] - bbox[0], bbox[2])
        bbox[3] = min(img.shape[2] - bbox[1], bbox[3])

        max_size = max(bbox[2], bbox[3])
        # pad the box to a square and crop from the image
        raw_center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
        new_left_top = [raw_center[0] - max_size / 2, raw_center[1] - max_size / 2]
        new_right_bottom = [raw_center[0] + max_size / 2, raw_center[1] + max_size / 2]
        # move the new bbox to keep the box in the image
        if new_left_top[0] < 0:
            new_right_bottom[0] -= new_left_top[0]
            new_left_top[0] = 0
        if new_left_top[1] < 0:
            new_right_bottom[1] -= new_left_top[1]
            new_left_top[1] = 0
        if new_right_bottom[0] > img.shape[1]:
            new_left_top[0] -= new_right_bottom[0] - img.shape[1]
            new_right_bottom[0] = img.shape[1]
        if new_right_bottom[1] > img.shape[2]:
            new_left_top[1] -= new_right_bottom[1] - img.shape[2]
            new_right_bottom[1] = img.shape[2]
        # convert the new bbox to the form of [top, left, height, width]
        bbox = [
            int(new_left_top[0]),
            int(new_left_top[1]),
            int(new_right_bottom[0] - new_left_top[0]),
            int(new_right_bottom[1] - new_left_top[1]),
        ]

        img = TF.crop(
            img.clone(), 
            *bbox,
        )

        return img, raw_mask

    def __len__(self):
        return len(self.meta_info)
    
    def __getitem__(self, index):
        item = self.meta_info[index]
        img_id = item["image"]["path"]

        # just for debug
        while not os.path.exists(os.path.join(self.data_path, img_id)):
            index = random.randint(0, len(self.meta_info) - 1)
            item = self.meta_info[index]
            img_id = item["image"]["path"]

        caption = item["image_caption"][0][-1]
        class_names = item["instance_caption"]
        
        random_instance_id = random.choice(list(range(len(class_names))))
        class_name = class_names[random_instance_id][-1]
        instance_id = class_names[random_instance_id][1]

        # print(class_names, class_name)
        # class_name = remove_a(class_name)

        box = item["annotations"][instance_id]["bbox"]
        # convert the box [x, y, w, h] to [x1, y1, x2, y2] and normalize to [0, 1]
        box = [
            box[0] / item["image"]["width"],
            box[1] / item["image"]["height"],
            (box[0] + box[2]) / item["image"]["width"],
            (box[1] + box[3]) / item["image"]["height"], 
        ]

        # box to tensor
        box = torch.tensor(box)

        seg = item["annotations"][instance_id]["segmentation"]
        
        img_path = os.path.join(self.data_path, img_id)
        image = Image.open(img_path).convert("RGB")
        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image = image.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        # convert the PIL.Image img to torch.Tensor
        image = TF.to_tensor(image)

        mask = torch.from_numpy(maskUtils.decode(seg)).bool()
        mask = mask.squeeze(0)
        processed_img, mask = self._process_img(image, box, mask)
        tokenized_class_name, tokenized_text, tokens_to_replace = \
            _find_tokens_to_replace(class_name, caption)

        crop = T.RandomCrop(self.crop_res)
        # flip = T.RandomHorizontalFlip(float(self.flip_prob))

        image = crop(image)
        mask = crop(mask)
        # flip image and mask
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            box = torch.tensor([1 - box[2], box[1], 1 - box[0], box[3]])
        image = 2 * image - 1
        
        # TODO: normalize the image & process the cropped image for CLIP embedding
        # to PIL image
        processed_img = TF.to_pil_image(processed_img, mode="RGB")
        processed_img = _transform(224)(processed_img)
        
        res = dict(
            img_id=img_id,
            instance_id=instance_id,
            edited=image,
            class_name=class_name,
            box=box,
            mask=mask,
            text=caption,
            tokens_to_replace=tokens_to_replace,
            processed_img=processed_img, # cropped image
            tokenized_class_name=tokenized_class_name,
            tokenized_text=tokenized_text,
        )
        # # print the type of values in res
        # for k, v in res.items():
        #     print(k, type(v))
        return res



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=100,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    opt = parser.parse_args()

    dist_init()
    seed_everything(opt.seed + get_rank())

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir


    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "source")
    result_path = os.path.join(outpath, "results")
    grid_path=os.path.join(outpath, "grid")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(grid_path, exist_ok=True)
  

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext

    dataset_obj = SAValDataset2(
        path="/mnt/external/datasets/sam/",
        split="val",
    )

    # split dataset into different ranks
    inds = list(range(len(dataset_obj)))
    inds = inds[get_rank()::get_world_size()]

    for idx in tqdm(inds):
        # image_path = ""
        # mask_path = ""
        # reference_path = ""
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    # filename=os.path.basename(image_path)
                    # img_p = Image.open(image_path).convert("RGB")
                    # image_tensor = get_tensor()(img_p)
                    # image_tensor = image_tensor.unsqueeze(0)
                    # ref_p = Image.open(reference_path).convert("RGB").resize((224,224))
                    # ref_tensor=get_tensor_clip()(ref_p)
                    # ref_tensor = ref_tensor.unsqueeze(0)
                    # mask=Image.open(mask_path).convert("L")
                    # mask = np.array(mask)[None,None]
                    # mask = 1 - mask.astype(np.float32)/255.0
                    # mask[mask < 0.5] = 0
                    # mask[mask >= 0.5] = 1
                    # mask_tensor = torch.from_numpy(mask)
                    image_tensor = dataset_obj[idx]['edited'][None, ...]
                    ref_tensor   = dataset_obj[idx]['processed_img'][None, ...]
                    mask_tensor  = 1 - dataset_obj[idx]['mask'][None, ...]
                    filename     = dataset_obj[idx]['img_id']
                    filename     = os.path.basename(filename)
                    # import pdb;pdb.set_trace()
                    inpaint_image = image_tensor*mask_tensor
                    test_model_kwargs={}
                    test_model_kwargs['inpaint_mask'] = mask_tensor.to(device)
                    test_model_kwargs['inpaint_image'] = inpaint_image.to(device)
                    ref_tensor=ref_tensor.to(device)
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.learnable_vector
                    c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
                    c = model.proj_out(c)
                    inpaint_mask=test_model_kwargs['inpaint_mask']
                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image']=z_inpaint
                    test_model_kwargs['inpaint_mask']=Resize([z_inpaint.shape[-2],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        test_model_kwargs=test_model_kwargs)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                    x_checked_image=x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    def un_norm(x):
                        return (x+1.0)/2.0
                    def un_norm_clip(x):
                        x[0,:,:] = x[0,:,:] * 0.26862954 + 0.48145466
                        x[1,:,:] = x[1,:,:] * 0.26130258 + 0.4578275
                        x[2,:,:] = x[2,:,:] * 0.27577711 + 0.40821073
                        return x

                    if not opt.skip_save:
                        for i,x_sample in enumerate(x_checked_image_torch):
                            

                            all_img=[]
                            all_img.append(un_norm(image_tensor[i]).cpu())
                            all_img.append(un_norm(inpaint_image[i]).cpu())
                            ref_img=ref_tensor
                            ref_img=Resize([opt.H, opt.W])(ref_img)
                            all_img.append(un_norm_clip(ref_img[i]).cpu())
                            all_img.append(x_sample)
                            grid = torch.stack(all_img, 0)
                            grid = make_grid(grid)
                            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                            img = Image.fromarray(grid.astype(np.uint8))
                            img = put_watermark(img, wm_encoder)
                            img.save(os.path.join(grid_path, 'grid-'+filename[:-4]+'_'+str(opt.seed)+'.png'))
                            
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img = put_watermark(img, wm_encoder)
                            img.save(os.path.join(result_path, filename[:-4]+'_'+str(opt.seed)+".png"))
                            
                            mask_save=255.*rearrange(un_norm(inpaint_mask[i]).cpu(), 'c h w -> h w c').numpy()
                            mask_save= cv2.cvtColor(mask_save,cv2.COLOR_GRAY2RGB)
                            mask_save = Image.fromarray(mask_save.astype(np.uint8))
                            mask_save.save(os.path.join(sample_path, filename[:-4]+'_'+str(opt.seed)+"_mask.png"))
                            GT_img=255.*rearrange(all_img[0], 'c h w -> h w c').numpy()
                            GT_img = Image.fromarray(GT_img.astype(np.uint8))
                            GT_img.save(os.path.join(sample_path, filename[:-4]+'_'+str(opt.seed)+"_GT.png"))
                            inpaint_img=255.*rearrange(all_img[1], 'c h w -> h w c').numpy()
                            inpaint_img = Image.fromarray(inpaint_img.astype(np.uint8))
                            inpaint_img.save(os.path.join(sample_path, filename[:-4]+'_'+str(opt.seed)+"_inpaint.png"))
                            ref_img=255.*rearrange(all_img[2], 'c h w -> h w c').numpy()
                            ref_img = Image.fromarray(ref_img.astype(np.uint8))
                            ref_img.save(os.path.join(sample_path, filename[:-4]+'_'+str(opt.seed)+"_ref.png"))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
