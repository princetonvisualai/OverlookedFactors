import pickle, time
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dtype = torch.float32

arch = 'resnet18'
model = torchvision.models.resnet18(pretrained=True)

center_crop = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


count = 0; start_time = time.time()

img_to_feature = {}

model_base = torch.nn.Sequential( *list(model.children())[:-1])
model_base.eval()
model_base = model_base.to(device)

names = pickle.load(open('test.pkl', 'rb'))

os.mkdir('cub_imagenet')

with torch.no_grad():
    for i in range(len(names)):
        img_name = names[i]['img_name'] 
        img = Image.open(img_name).convert('RGB')
        img = center_crop(img)
        img = img.to(device=device, dtype = dtype)
        img = img.unsqueeze(0)
        
        feat = model_base.forward(img).detach().cpu().numpy()
        
        img_to_feature[img_name] = feat
        
        count += 1
        if count%1000 == 0: 
            elapsed_time = (time.time() - start_time)/60
            print('Processed {} images in {:.2f} minutes'.format(count, elapsed_time), flush=True)

        pickle.dump(img_to_feature, open('cub_imagenet/imagenet_features.pkl'.format(split), 'wb+'))

