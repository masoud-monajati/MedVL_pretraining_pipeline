## Medical Vision-Language Pre-Training for Brain Abnormalities

This is the repository for our paper on building a pipeline for pre-training medical Vision-Language models.

Note: the repository is mainly adapted from the <a href="https://github.com/salesforce/BLIP">BLIP</a> repository.

To install the dependencies, run <pre/>pip install -r requirements.txt</pre> 

### Dataset:
You may use scrape.py to collect the data from PubMed resources with a proper keyword and number of data. You may also reach out to us for sharing the data discussed in the paper.
We used <a href="https://cloud.google.com/vision/docs/ocr">Google OCR</a> and an NLP tool from <a href="https://cloud.google.com/vision/docs/ocr">This paper</a> to preprocess image-caption data into subfigure/subcaption data stored <a href="">here</a>. 


### Pre-train:
Following the BLIP GitHub repository, here are the steps for pre-training.
1. Prepare training json files where each json file contains a list. Each item in the list is a dictonary with two key-value pairs: {'image': path_of_image, 'caption': text_of_image}. 
2. In configs/pretrain.yaml, set 'train_file' as the paths for the json files .
3. Pre-train the model using 8 A100 GPUs:
<pre>python -m torch.distributed.run --nproc_per_node=8 pretrain.py --config ./configs/Pretrain.yaml --output_dir output/Pretrain </pre> 

### Image-Text Retrieval evaluation:
Following the BLIP GitHub repository, here are the steps for evaluation and fine-tuning.

To evaluate, run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco \
--evaluate</pre> 
To fine-tune the pre-trained checkpoint using 8 A100 GPUs, first set 'pretrained' in configs/retrieval_coco.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco </pre> 

### Visualization:
To visualize the attention map, you may use vis.py and visualize the attention map of the subfigure with respect to the subcaption.

### Acknowledgement
The implementation of this repo relies on resources from <a href="https://github.com/salesforce/BLIP">BLIP</a>, <a href="https://github.com/salesforce/ALBEF">ALBEF</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a>.
