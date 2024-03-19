from google.cloud import vision
import io
import json
from PIL import Image
from tqdm import tqdm

client = vision.ImageAnnotatorClient()

med_path='/home/monajati/main/med/blip/med_data/medical_20k.json'
med_data=json.load(open(med_path,'r'))

for j in tqdm(range(len(med_data))):
        
    path='/home/monajati/main/med/blip/images/'+med_data[j]['image'][51:]

    #sub_id=ann['Sub_ID']

    ori_image = Image.open(path).convert('RGB')

    #print(ann['Img_id'])

    #test_path='0.png'

    crop_path='/home/monajati/main/med/blip/med_data/bound/bounding_boxes_new/'+med_data[j]['image'][51:]+'.json'
    
    with open(crop_path, "r") as jsonFile:
        crop_data = json.load(jsonFile)

    #data["location"] = "NewPath"
    
    #crop_data=json.load(open(crop_path,'r'))
    #jsonFile.close()
    for i in range(len(crop_data)):
        left = crop_data[i]['x']
        top = crop_data[i]['y']
        bottom = crop_data[i]['h'] + top
        right = crop_data[i]['w'] + left
        
        im1 = ori_image.crop((left, top, right, bottom))
        im2=im1

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
        
        crop_data[i]['OCR']=char_list
    
    with open(crop_path, "w") as jsonFile:
        json.dump(crop_data, jsonFile)    
      
    
    #image = self.transform(im2)
    #caption = pre_caption(ann['text'],500)

print("done")