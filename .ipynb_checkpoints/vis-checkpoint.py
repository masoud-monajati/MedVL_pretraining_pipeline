import sys

model_path = sys.argv[1] #'../VL/Example/refcoco.pth'
bert_config_path = sys.argv[2] #'configs/config_bert.json'
image_size = 224


from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.blip_itm import blip_itm

def load_demo_image(image_size,device):
    raw_image = Image.open(sys.argv[3]).convert('RGB') 
    
    image_path = sys.argv[3]
    
    crop_path='/home/monajati/main/med/blip/med_data/bounding_boxes_20k/'+image_path[36:]+'.json'

    if len(json.load(open(crop_path,'r')))==0:
        raw_image = raw_image
    else:
        left = json.load(open(crop_path,'r'))[0]['x']
        top = json.load(open(crop_path,'r'))[0]['y']
        bottom = json.load(open(crop_path,'r'))[0]['h'] + top
        right = json.load(open(crop_path,'r'))[0]['w'] + left

        raw_image = raw_image.crop((left, top, right, bottom))

    w,h = raw_image.size
    #display(raw_image.resize((w//5,h//5)))
    raw_image.save("test.png")
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)
    
    return image


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
image = load_demo_image(image_size=image_size, device=device)


caption_path='/home/monajati/main/med/blip/med_data/medical_20k.json'
h=json.load(open(caption_path,'r'))

print("name",sys.argv[3][36:-4])

indx=int(sys.argv[3][36:-4])

caption = pre_caption(h[indx]['caption'])

print("caption",caption)

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
    ax[i+1].imshow(gradcam_image)
    ax[i+1].set_yticks([])
    ax[i+1].set_xticks([])
    ax[i+1].set_xlabel(word)
plt.savefig('output.pdf')
