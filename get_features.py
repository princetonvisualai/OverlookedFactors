import pickle, time
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

import numpy as np
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dtype = torch.float32

arch = 'resnet18'

model_file = '%s_places365.pth.tar' % arch
places_model = torchvision.models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
places_model.load_state_dict(state_dict)
places_model.eval()

places_model_base = torch.nn.Sequential( *list(places_model.children())[:-1])
places_model_base.eval()


model = torchvision.models.resnet18(pretrained=True)
model.eval()
center_crop = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
model_base = torch.nn.Sequential( *list(model.children())[:-1])
model_base.eval()

if not os.path.exists('ADE20k'):
    os.mkdir('ADE20k')

A = pickle.load(open('ade20k_imagelabels.pkl', 'rb'))
for name, m in [('features', places_model_base), ('logits', places_model), ('imagenet', model_base)]:
    with torch.no_grad():
        
        m = m.to(device)

        for split in ['train', 'val', 'test']:
            count = 0; start_time = time.time()
            img_to_scene = {}  
            img_to_feature = {}
            img_names = A[split]
            for img_name in img_names: 
                img_path = img_name
                img = Image.open(img_path).convert('RGB')
                img = center_crop(img)
                img = img.to(device=device, dtype = dtype)
                img = img.unsqueeze(0)
               
                sc = m(img)
                img_to_feature[img_path] = sc.detach().cpu().numpy().squeeze() 
                
                if name=='logits':
                    h_x = torch.nn.functional.softmax(sc, 1).data.squeeze()
                    probs, idx = h_x.sort(0, True)
                
                    scene = int(idx[0].data.cpu().numpy())
                    img_to_scene[img_path] = scene
                

                count += 1
                if count%1000 == 0: 
                    elapsed_time = (time.time() - start_time)/60
                    print('Processed {} images in {:.2f} minutes'.format(count, elapsed_time), flush=True)
                    
            pickle.dump(img_to_feature, open('ADE20k/{}_{}.pkl'.format(split, name), 'wb+'))
            if name=='logits':
                pickle.dump(img_to_scene, open('ADE20k/{}_scene.pkl'.format(split), 'wb+'))

if not os.path.exists('Pascal'):
    os.mkdir('Pascal')

A = pickle.load(open('pascal_imagelabels.pkl', 'rb'))
for name, m in [('features', places_model_base), ('logits', places_model)]:
    with torch.no_grad():
        
        m = m.to(device)

        for split in ['train', 'val', 'test']:
            count = 0; start_time = time.time()
            img_to_scene = {}  
            img_to_feature = {}
            img_names = A[split]
            for img_name in img_names: 
                img_path = img_name
                img = Image.open(img_path).convert('RGB')
                img = center_crop(img)
                img = img.to(device=device, dtype = dtype)
                img = img.unsqueeze(0)
               
                sc = m(img)
                img_to_feature[img_path] = sc.detach().cpu().numpy().squeeze() 
                
                if name=='logits':
                    h_x = torch.nn.functional.softmax(sc, 1).data.squeeze()
                    probs, idx = h_x.sort(0, True)
                
                    scene = int(idx[0].data.cpu().numpy())
                    img_to_scene[img_path] = scene
                

                count += 1
                if count%1000 == 0: 
                    elapsed_time = (time.time() - start_time)/60
                    print('Processed {} images in {:.2f} minutes'.format(count, elapsed_time), flush=True)
                    
            pickle.dump(img_to_feature, open('Pascal/{}_{}.pkl'.format(split, name), 'wb+'))
            if name=='logits':
                pickle.dump(img_to_scene, open('Pascal/{}_scene.pkl'.format(split), 'wb+'))




