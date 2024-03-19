import sys

#model_path = sys.argv[1] #'../VL/Example/refcoco.pth'
model_path = '/data1/monajati/med/checkpoints/pretrain3_75/checkpoint_54.pth'
#bert_config_path = sys.argv[2] #'configs/config_bert.json'
bert_config_path = '/configs/bert_config.json'
image_size = 224


from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.blip_itm import blip_itm

# model loading

model = blip_itm(pretrained=model_path, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

import re

def pre_caption(caption,max_words=200):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])            
    return caption

from PIL import Image

import cv2
import numpy as np

from skimage import transform as skimage_transform
from scipy.ndimage import filters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

def getAttMap(img, attMap, blur = True, overlap = True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap

# input preprocessing
#image = load_demo_image(image_size=image_size, device=device)

med_path='/home/monajati/main/med/blip/med_data/med_captions_processed.json'

med_data=json.load(open(med_path,'r'))

data=med_data[14]

print('data',data)

im=data['Img_id']
sub_id=data['Sub_ID']
caption=data['text']
'''
Img_id:"0.png"
Sub_ID:"a"
text
'''

image_path=path='/home/monajati/main/med/blip/images/'+im

ori_image = Image.open(image_path).convert('RGB') 

#image_path = sys.argv[3]
crop_path='/home/monajati/main/med/blip/med_data/bound/bounding_boxes_new/'+im+'.json'

#crop_path='/home/monajati/main/med/blip/med_data/bound/bounding_boxes_new/'+image_path[36:]+'.json'

'''
if len(json.load(open(crop_path,'r')))==1:
    raw_image = raw_image
else:
    left = json.load(open(crop_path,'r'))[0]['x']
    top = json.load(open(crop_path,'r'))[0]['y']
    bottom = json.load(open(crop_path,'r'))[0]['h'] + top
    right = json.load(open(crop_path,'r'))[0]['w'] + left

    raw_image = raw_image.crop((left, top, right, bottom))
'''
if sub_id!="":
    print('crop_path length',len(json.load(open(crop_path,'r'))))
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
            '''

            img_byte_arr = io.BytesIO()
            im1.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            image = vision.Image(content=img_byte_arr)

            response = client.text_detection(image=image)
            texts = response.text_annotations

            #print("texts",texts[0].description[0])
            #print('Texts:')
            char_list=[]
            for text in texts:
                #print('\n"{}"'.format(text.description))
                #print(text.description)
                char_list.append(text.description.lower())
            '''
            char_list=json.load(open(crop_path,'r'))[i]['OCR']
            print('char_list',char_list)

            if sub_id in char_list:
                flag=1
                break
        '''
        try:
            if sub_id not in char_list:
                im2=ori_image
        except:
            print("image_id",ann['Img_id'])
        '''
        if flag==0:
            im2=ori_image

else:
    im2=ori_image
#im1.save("102_1.png")

#print('flag',flag)

'''
try:
    image = self.transform(im2)
except:
    print("image_id",ann['Img_id'])
'''
#image = self.transform(im2)
#caption = pre_caption(ann['text'],500)

raw_image=im2
    
w,h = raw_image.size
#display(raw_image.resize((w//5,h//5)))
raw_image.save("test.png")
transform = transforms.Compose([
    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 
image = transform(raw_image).unsqueeze(0).to(device)

'''
caption_path='/home/monajati/main/med/blip/med_data/medical_20k.json'
h=json.load(open(caption_path,'r'))

print("name",sys.argv[3][36:-4])

indx=int(sys.argv[3][36:-4])
'''

caption = pre_caption(caption)

#print("caption",caption)

#caption = open(sys.argv[4]).readlines()[0].strip()
#caption = pre_caption(caption)


# modeling
block_num = 8
model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True

itm_output = model(image,caption,match_head='itm')
itm_score = -torch.log(torch.nn.functional.softmax(itm_output,dim=1)[:,1])
loss = itm_score.sum()
model.zero_grad()
loss.backward()

#text_input = tokenizer(caption, return_tensors="pt")
text_input = model.tokenizer(caption, padding='max_length', truncation=True, max_length=200, 
                      return_tensors="pt").to(image.device) 

#print("text_input",text_input)
with torch.no_grad():
    mask = text_input.attention_mask.view(text_input.attention_mask.size(0),1,-1,1,1)

    grads=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients()
    cams=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map()

    #cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, 24, 24) * mask
    #grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, 24, 24) * mask
    cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, 14, 14) * mask
    grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, 14, 14) * mask

    gradcam = cams * grads
    gradcam = gradcam[0].mean(0).cpu().detach()

num_image = 50 #len(text_input.input_ids[0]) 
fig, ax = plt.subplots(num_image, 1, figsize=(15,5*num_image))

rgb_image = cv2.imread("/home/monajati/main/med/blip/test.png")[:, :, ::-1]
rgb_image = np.float32(rgb_image) / 255

print("reading image is done")

ax[0].imshow(rgb_image)
ax[0].set_yticks([])
ax[0].set_xticks([])
ax[0].set_xlabel("Image")
            
#for i,token_id in enumerate(text_input.input_ids[0][1:]):
for i,token_id in enumerate(text_input.input_ids[0][1:num_image]):
    word = model.tokenizer.decode([token_id])
    gradcam_image = getAttMap(rgb_image, gradcam[i+1])
    
    print('word',word)
    if i==5:
        background_gen=gradcam_image
    if 6<=i<9:
        background_gen+=gradcam_image
    ax[i+1].imshow(gradcam_image)
    ax[i+1].set_yticks([])
    ax[i+1].set_xticks([])
    ax[i+1].set_xlabel(word)
ax[i].imshow(background_gen)
ax[i].set_yticks([])
ax[i].set_xticks([])
ax[i].set_xlabel("aneurysm sum")
ax[i+1].imshow(background_gen/4)
ax[i+1].set_yticks([])
ax[i+1].set_xticks([])
ax[i+1].set_xlabel("aneurysm avg")
plt.savefig('output_ori.pdf')
