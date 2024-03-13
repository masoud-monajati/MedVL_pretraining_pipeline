import requests
import json
import urllib

output_dir='data'
query='chest x-ray'
total_num = 20000

cnt = 0
out_caption = []
query = '%20'.join(query.split())
while cnt < total_num:
    URL=f'https://openi.nlm.nih.gov/api/search?at=cr&coll=pmc&m={cnt+1}&n={cnt+100}&query={query}'
    page = requests.get(URL)
    page = json.loads(page.text)
    ic_list = page['list']
    for ic in ic_list:
        image_url = 'https://openi.nlm.nih.gov' + ic['imgLarge']
        #urllib.request.urlretrieve(image_url, f'{output_dir}/{cnt}.png')
        caption = ic['image']['caption']
        out_caption.append({'caption': caption, 'img_file': f'{cnt}.png', 'img_url': image_url})
        cnt += 1
    if cnt > page['total']:
        break

json.dump(out_caption, open(f'{output_dir}/caption.json', 'w'), indent=4)
