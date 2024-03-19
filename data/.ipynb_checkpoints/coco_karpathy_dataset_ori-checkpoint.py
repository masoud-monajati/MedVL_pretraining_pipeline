import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

class coco_karpathy_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        #url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        #filename = 'coco_karpathy_train.json'

        #download_url(url,ann_root)
        
        #self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))

        #filename = 'blip_downstream_train.json'
        filename = 'train.json'
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        #self.image_root = image_root
        self.max_words = 200 #max_words      
        self.prompt = '' #prompt
        
          
        '''
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            #img_id = ann['image_id']
            img_id = ann['Img_id'][:-4]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1  
        '''
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index): 
        
        ann = self.annotation[index]
        
        #image_path = os.path.join(self.image_root,ann['image']) 
        image_path = '/home/monajati/main/med/blip/images/'+ann['image'][51:]
        #print(image_path)
        #print(begh)
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words)
        return image, caption, self.img_ids[ann['image_id']]
        '''
        ann = self.annotation[index]
        
        #image_path = ann['image'] #os.path.join(self.image_root,ann['image'])   
        #image_path = '/home/monajati/main/med/blip/images/'+ann['image'][51:]
        image_path = '/home/monajati/main/med/blip/images/'+ann['Img_id']
        
        image = Image.open(image_path).convert('RGB')   
        
        #crop_path='/home/monajati/main/med/blip/med_data/bounding_boxes_20k/'+ann['image'][51:]+'.json'
        crop_path='/home/monajati/main/med/blip/med_data/bounding_boxes_20k/'+ann['Img_id']+'.json'
        
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
        #image = self.transform(image)
        
        #caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        caption = self.prompt+pre_caption(ann['text'], self.max_words) 
        

        #return image, caption, self.img_ids[ann['image_id']]
        return image, caption, self.img_ids[ann['Img_id'][:-4]]
        '''
    
    
    
class coco_karpathy_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        #urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
        #        'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        #filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        #download_url(urls[split],ann_root)
        
        #filenames = {'val': 'train_blip_caption_val.json', 'test': 'train_blip_caption_test.json'}
        filenames = {'val': 'val.json', 'test': 'test.json'}
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = ann['image'] #os.path.join(self.image_root,ann['image'])  
        image_path = '/home/monajati/main/med/blip/images/'+ann['image'][51:]
        image = Image.open(image_path).convert('RGB')
        
        crop_path='/home/monajati/main/med/blip/med_data/bounding_boxes_20k/'+ann['image'][51:]+'.json'
        
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
        
        #image = self.transform(image)          
        
        img_id = ann['image_id'] #.split('/')[-1].strip('.jpg').split('_')[-1]
        
        return image, int(img_id)   
    
    
class coco_karpathy_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        
        max_words=200
        #filenames = {'val': 'blip_downstream_val.json', 'test': 'blip_downstream_test.json'}
        filenames = {'val': 'val.json', 'test': 'test.json'}
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['Img_id'][:-4])
            self.img2txt[img_id] = []
            #caption = ann['caption']
            caption = ann['text']
            self.text.append(pre_caption(caption,max_words))
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1#
                                    
        
        '''
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        download_url(urls[split],ann_root)
        '''
        filenames = {'val': 'val.json', 'test': 'test.json'}
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            #print(ann['image'])
            #print(begh)
            self.img2txt[img_id] = []
            caption = ann['caption']
            self.text.append(pre_caption(caption,max_words))
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1#
            '''
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1#
            '''
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        #image_path = os.path.join(self.image_root, self.annotation[index]['image'])
        
        ann = self.annotation[index]
        #image_path = ann['image'] #os.path.join(self.image_root,ann['image'])
        image_path = '/home/monajati/main/med/blip/images/'+ann['image'][51:]
        #image_path = '/home/monajati/main/med/blip/images/'+ann['Img_id']
        image = Image.open(image_path).convert('RGB') 
        '''
        crop_path='/home/monajati/main/med/blip/med_data/bounding_boxes_20k/'+ann['Img_id']+'.json'
        
        if len(json.load(open(crop_path,'r')))==0:
            im1 = image
        else:
            left = json.load(open(crop_path,'r'))[0]['x']
            top = json.load(open(crop_path,'r'))[0]['y']
            bottom = json.load(open(crop_path,'r'))[0]['h'] + top
            right = json.load(open(crop_path,'r'))[0]['w'] + left
        
            im1 = image.crop((left, top, right, bottom))
        
        #im1.save("102_1.png")
        '''
        
        #print(begh)
        
        image = self.transform(image)
        
        
        #image = self.transform(image)  

        return image, index