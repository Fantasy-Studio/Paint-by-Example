import os
from PIL import Image
import torchvision
from tqdm import tqdm
trans=torchvision.transforms.Compose([torchvision.transforms.Resize(512),torchvision.transforms.CenterCrop(512)])
src_dir='dataset/coco/test2017'
tag_dir='test_bench/test_set_GT'
os.makedirs(tag_dir, exist_ok=True)
for file in tqdm(os.listdir(src_dir)):
    path=os.path.join(src_dir,file)
    img=Image.open(path)
    img=trans(img)
    img.save(os.path.join(tag_dir,file[:-4]+'.png'))