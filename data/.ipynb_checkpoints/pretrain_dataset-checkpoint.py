import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from data.utils import pre_caption
import os,glob

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, laion_path, transform): 

        self.ann_pretrain = []
        for f in ann_file:
            print('loading '+f)
            ann = json.load(open(f,'r'))
            self.ann_pretrain += ann
        
        self.laion_path = laion_path
        if self.laion_path:
            self.laion_files = glob.glob(os.path.join(laion_path,'*.json'))

            print('loading '+self.laion_files[0])
            with open(self.laion_files[0],'r') as f:
                self.ann_laion = json.load(f)  

            self.annotation = self.ann_pretrain + self.ann_laion
        else:
            self.annotation = self.ann_pretrain
            
        self.transform = transform


    def reload_laion(self, epoch):
        n = epoch%len(self.laion_files)
        print('loading '+self.laion_files[n])
        with open(self.laion_files[n],'r') as f:
            self.ann_laion = json.load(f)      
        
        self.annotation = self.ann_pretrain + self.ann_laion    
        
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
          
        
        #test_path='102.png'
        ann = self.annotation[index]
        
        path='/home/monajati/main/med/blip/images/'+ann['image'][51:]
        #print('path',ann['image'][51:])
        
        
        image = Image.open(path).convert('RGB')
        
        #test_path='0.png'
        
        crop_path='/home/monajati/main/med/blip/med_data/bounding_boxes_20k/'+ann['image'][51:]+'.json'
        #print('crop_path',ann['image'][51:])
        #print(begh)
        if len(json.load(open(crop_path,'r')))==0:
            im1 = image
        else:
            left = json.load(open(crop_path,'r'))[0]['x']
            top = json.load(open(crop_path,'r'))[0]['y']
            bottom = json.load(open(crop_path,'r'))[0]['h'] + top
            right = json.load(open(crop_path,'r'))[0]['w'] + left
        
            im1 = image.crop((left, top, right, bottom))
        
        #im1.save("102_1.png")
        
        #print(begh)
        
        image = self.transform(im1)
        caption = pre_caption(ann['caption'],500)
        
        #print("caption",caption)
        #print(begh)
        
        return image, caption