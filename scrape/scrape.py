import requests
import json
import urllib
import re

output_dir='data'
query='brain'
total_num = 1000

cnt = 0
out_caption = []
query = '%20'.join(query.split())
paper_url=[]
journal_title=[]
journal_date=[]
while cnt < total_num:
    URL=f'https://openi.nlm.nih.gov/api/search?at=cr&coll=pmc&m={cnt+1}&n={cnt+100}&query={query}'
    page = requests.get(URL)
    page = json.loads(page.text)
    print('page',page['list'][0])
    print("+++++++")
    print('page',page['list'][1])
    print(begh)
    #print('page.keys()',page.keys())
    #print('page.keys()',page['min'], page['max'], page['count'], page['total'], page['approximage'])
    #print(len(page['list']))
    #print(page['list'][0])
    ic_list = page['list']
    for ic in ic_list:
        print(cnt)
        if ic['pubMed_url'] not in paper_url:
            paper_url.append(ic['pubMed_url'])     
        if ic['journal_title'] not in journal_title:
            journal_title.append(ic['journal_title'])
        
        
        pattern = r'\b\d{4}\b'
        
        
        if 'journal_date' in ic.keys():
            matches = re.findall(pattern, ic['journal_date']['year'])
            journal_date.append(int(matches[0]))
        image_url = 'https://openi.nlm.nih.gov' + ic['imgLarge']
        #urllib.request.urlretrieve(image_url, f'{output_dir}/{cnt}.png')
        caption = ic['image']['caption']
        out_caption.append({'caption': caption, 'img_file': f'{cnt}.png', 'img_url': image_url})
        cnt += 1
    if cnt > page['total']:
        break
print(journal_date)
print(len(paper_url), len(journal_title))
print(journal_title)

print(min(journal_date), max(journal_date))
json.dump(out_caption, open(f'{output_dir}/caption2.json', 'w'), indent=4)
