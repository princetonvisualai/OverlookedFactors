import csv
import pandas as pd
import os
from sklearn.model_selection import train_test_split

image_df = pd.read_csv('dataset/broden1_224/index.csv')

for idx in image_df.index:
    if (image_df['image'][idx]).split('/')[0]!='ade20k':
        continue
    full_image_name = 'dataset/broden1_224/images/{}'.format(image_df['image'][idx])
    #print(idx)
    images.append(full_image_name)
    labels[full_image_name] = []

    for cat in ['object', 'part']:
        if image_df[cat].notnull()[idx]:
            for x in image_df[cat][idx].split(';'):    
                img_labels = Image.open('dataset/broden1_224/images/{}'.format(x))
                numpy_val = np.array(img_labels)[:, :, 0]+ 256* np.array(img_labels)[:, :, 1]
                code_val = [i for i in np.sort(np.unique(numpy_val))[1:]]
                labels[full_image_name] += code_val



images_train, images_valtest = train_test_split(images, test_size=0.4, random_state=42)
images_val, images_test = train_test_split(images_valtest, test_size=0.5, random_state=42)

with open('ade20k_imagelabels.pkl', 'wb+') as handle:
    pickle.dump({'train': images_train, 'val':images_val, 'test':images_test, 'labels':labels}, handle)

for idx in image_df.index:
    if (image_df['image'][idx]).split('/')[0]!='pascal':
        continue
    full_image_name = 'dataset/broden1_224/images/{}'.format(image_df['image'][idx])
    #print(idx)
    images.append(full_image_name)
    labels[full_image_name] = []

    for cat in ['object', 'part']:
        if image_df[cat].notnull()[idx]:
            for x in image_df[cat][idx].split(';'):    
                img_labels = Image.open('dataset/broden1_224/images/{}'.format(x))
                numpy_val = np.array(img_labels)[:, :, 0]+ 256* np.array(img_labels)[:, :, 1]
                code_val = [i for i in np.sort(np.unique(numpy_val))[1:]]
                labels[full_image_name] += code_val



images_train, images_valtest = train_test_split(images, test_size=0.4, random_state=42)
images_val, images_test = train_test_split(images_valtest, test_size=0.5, random_state=42)

with open('pascal_imagelabels.pkl', 'wb+') as handle:
    pickle.dump({'train': images_train, 'val':images_val, 'test':images_test, 'labels':labels}, handle)




