import json
import os
import random
#import io
#from google.cloud import vision
#import time

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
        
        path='/home/monajati/main/med/blip/images/'+ann['Img_id']
        
        sub_id=ann['Sub_ID']
        
        ori_image = Image.open(path).convert('RGB')
        
        #print(ann['Img_id'])
        
        #test_path='0.png'
        
        crop_path='/home/monajati/main/med/blip/med_data/bound/bounding_boxes_new/'+ann['Img_id']+'.json'
        if sub_id!="":
            if len(json.load(open(crop_path,'r')))==1:
                im2 = ori_image
            else:
                #client = vision.ImageAnnotatorClient()
                #char_list=[]
                flag=0
                for i in range(len(json.load(open(crop_path,'r')))):
                    #time.sleep(0.1)
                    left = json.load(open(crop_path,'r'))[i]['x']
                    top = json.load(open(crop_path,'r'))[i]['y']
                    bottom = json.load(open(crop_path,'r'))[i]['h'] + top
                    right = json.load(open(crop_path,'r'))[i]['w'] + left
                    
                    im1 = ori_image.crop((left, top, right, bottom))
                    im2=im1
                    char_list=json.load(open(crop_path,'r'))[i]['OCR']

                    if sub_id in char_list:
                        flag=1
                        break
                if flag==0:
                    im2=ori_image
                        
        else:
            im2=ori_image
        #im1.save("102_1.png")
        #im2=ori_image
        #print(begh)
        
        '''
        try:
            image = self.transform(im2)
        except:
            print("image_id",ann['Img_id'])
        '''
        image = self.transform(im2)
        caption = pre_caption(ann['text'],300)
        
        #print("caption",caption)
        #print(begh)
        
        return image, caption