from google.cloud import vision
import io
import json
from PIL import Image
from tqdm import tqdm

client = vision.ImageAnnotatorClient()

med_path='/home/monajati/main/med/blip/med_data/medical_20k.json'
med_data=json.load(open(med_path,'r'))

for j in tqdm(range(2)):
        
    path='/home/monajati/main/med/blip/images/'+med_data[j]['image'][51:]

    #sub_id=ann['Sub_ID']

    ori_image = Image.open(path).convert('RGB')

    #print(ann['Img_id'])

    #test_path='0.png'

    crop_path='/home/monajati/main/med/blip/med_data/OCR_test/'+med_data[j]['image'][51:]+'.json'
    
    with open(crop_path, "r") as jsonFile:
        crop_data = json.load(jsonFile)

    #data["location"] = "NewPath"
    
    #crop_data=json.load(open(crop_path,'r'))
    #jsonFile.close()
    print(crop_data[0]['OCR'])
    if 'b' in crop_data[0]['OCR']:
        print('yes')
       
      
    
    #image = self.transform(im2)
    #caption = pre_caption(ann['text'],500)

print("done")