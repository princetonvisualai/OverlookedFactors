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
                    
            pickle.dump(img_to_feature, open('ADE20k/{}_{}.pkl'.format(split, name), 'wb+'))
            if name=='logits':
                pickle.dump(img_to_scene, open('ADE20k/{}_scene.pkl'.format(split), 'wb+'))




count = 0; start_time = time.time()
img_to_scene = {}; img_to_scenegroup = {}; img_to_feature = {}

model_base = model_base.to(device)
#val_feat = []
#val_scenegroup = pickle.load(open('record/resnet18/val_scenegroup.pkl', 'rb'))
#val_scene = pickle.load(open('record/resnet18/val_scene.pkl', 'rb'))
#y_valgroup = np.zeros((len(val_scenegroup), 18)) 
#y_val = np.zeros(len(val_scenegroup)) 


pascal_images = os.listdir('../NetDissect-Lite/dataset/broden1_224/images/pascal')
        

with torch.no_grad():
    for split in ['train', 'val', 'test']:
    #for split, img_names in [('train', train_images), ('val', val_images)]:
    #for split in ['full']:
        
        count = 0; start_time = time.time()
        img_to_scene = {}; img_to_scenegroup = {}; 
        img_to_feature = {}
        img_to_logit = {} 
        img_names = A[split]
        for img_name in img_names: #pascal_images: #l in lines :
            #if img_name.endswith('.png'):
            #    continue
            #s = l.strip().split()
            img_path = img_name #'../NetDissect-Lite/dataset/broden1_224/images/pascal/{}'.format(img_name)
            img = Image.open(img_path).convert('RGB')
            img = center_crop(img)
            img = img.to(device=device, dtype = dtype)
            img = img.unsqueeze(0)
            
            logit = model.forward(img)
            feat = model_base.forward(img).detach().cpu().numpy()
            #if count==0:
            #    print(feat.shape)
            
            h_x = torch.nn.functional.softmax(logit, 1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            
            scene = int(idx[0].data.cpu().numpy())
            img_to_scene[img_path] = scene
            
            #scenegroup = sceneidx_to_scenegroupidx[scene]
            #img_to_scenegroup[img_path] = scenegroup
            
            img_to_feature[img_path] = feat
            #train_feat.append(feat)
            #y_traingroup[count, train_scenegroup[img_path]] = 1
            #y_train[count] = train_scene[img_path]
            img_to_logit[img_name] = h_x.cpu().numpy()

            count += 1
            if count%1000 == 0: 
                elapsed_time = (time.time() - start_time)/60
                print('Processed {} images in {:.2f} minutes'.format(count, elapsed_time), flush=True)
                
            #if count%10000==0:    
            #    pickle.dump(img_to_feature, open('record/COCO/{}_features77_{}.pkl'.format(split, count//10000), 'wb+'))
            #    img_to_feature = {}
        pickle.dump(img_to_scene, open('record/ADE20k_imagenet/{}_scene.pkl'.format(split), 'wb+'))
        #pickle.dump(img_to_scenegroup, open('record/ADE20k_imagenet/{}_scenegroup.pkl'.format(split), 'wb+'))
        pickle.dump(img_to_feature, open('record/ADE20k_imagenet/{}_features.pkl'.format(split), 'wb+'))
        pickle.dump(img_to_logit, open('record/ADE20k_imagenet/{}_logits.pkl'.format(split), 'wb+'))

#X_train = np.concatenate(train_feat)
#pickle.dump(X_train, open('record/resnet18_imagenet/cocotrain_features.pkl', 'wb+'))
#pickle.dump({'scene':y_train, 'scenegroup':y_traingroup}, open('record/resnet18_imagenet/train_scenes_np.pkl', 'wb+'))
#print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

"""

X_train = pickle.load(open('record/resnet18_imagenet/cocotrain_features.pkl','rb'))
X_val = pickle.load(open('record/resnet18_imagenet/cocoval_features.pkl','rb'))
y_train_all = pickle.load(open('record/resnet18_imagenet/train_scenes_np.pkl','rb'))['scenegroup']
y_val_all = pickle.load(open('record/resnet18_imagenet/val_scenes_np.pkl','rb'))['scenegroup']

y_train= np.zeros(y_train_all.shape[0])
y_val = np.zeros(y_val_all.shape[0])
for i in range(y_train.shape[0]):
    y_train[i] = np.nonzero(y_train_all[i])[0][0]
for i in range(y_val.shape[0]):
    y_val[i] = np.nonzero(y_val_all[i])[0][0]
#tree = DecisionTreeClassifier()     #MultiOutputClassifier(DecisionTreeClassifier())
#tree.fit(X_train, y_train)
clf = LogisticRegression(multi_class='multinomial', solver='sag')
clf.fit(X_train, y_train)
print(clf.score(X_val, y_val))
#print(hamming(tree.predict(X_val).astype(bool), y_val.astype(bool)))

img_path = '../../data/ILSVRC2012_img_val/'
val_feat = {}

for i in range(args.start+1, args.start+5001):
    name = img_path+'ILSVRC2012_val_{:08d}.JPEG'.format(i)
    img = center_crop(Image.open(name).convert('RGB')).reshape(1, 3, 224, 224)

    feat = model(img.to(device))
     
    val_feat[name] = np.argmax(feat.squeeze().detach().cpu().numpy())
    if i%500==0:
        print(i)
        with open('record/imagenet_val_features/predictions{}.pkl'.format(i), 'wb+') as handle:
            pickle.dump(val_feat, handle)
        val_feat = {}
"""
