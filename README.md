## Medical Vision-Language Pre-Training for Brain Abnormalities

This is the repository for our paper on building a pipeline for pre-training medical Vision-Language models.

Note: the repository is mainly adapted from the <a href="https://github.com/salesforce/BLIP">BLIP</a> repository.

To install the dependencies, run <pre/>pip install -r requirements.txt</pre> 

### Dataset:
for sharing the data, please reach out to us.

We used <a href="https://cloud.google.com/vision/docs/ocr">Google OCR</a> and an NLP tool from <a href="https://cloud.google.com/vision/docs/ocr">This paper</a> to preprocess image-caption data into subfigure/subcaption data stored <a href="">here</a>. 

You may use scrape.py to collect the data from PubMed resources with a proper keyword and number of data.

### Pre-train:
1. Prepare training json files where each json file contains a list. Each item in the list is a dictonary with two key-value pairs: {'image': path_of_image, 'caption': text_of_image}. 
2. In configs/pretrain.yaml, set 'train_file' as the paths for the json files .
3. Pre-train the model using 8 A100 GPUs:
<pre>python -m torch.distributed.run --nproc_per_node=8 pretrain.py --config ./configs/Pretrain.yaml --output_dir output/Pretrain </pre> 

### Image-Text Retrieval:
To evaluate, run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco \
--evaluate</pre> 
To finetune the pre-trained checkpoint using 8 A100 GPUs, first set 'pretrained' in configs/retrieval_coco.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco </pre> 

### Visualization:
To visualize the attention map, you may use vis.py for any data from PubMed.


### Citation
<pre>
@misc{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, 
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      eprint={2201.12086},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}</pre>

### Acknowledgement
The implementation of this repo relies on resources from <a href="https://github.com/salesforce/BLIP">BLIP</a>, <a href="https://github.com/salesforce/ALBEF">ALBEF</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a>.
