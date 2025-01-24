# KEEN

Pytorch Implementation of KEEN (kcat Prediction End-to-End Network with Pre-trained Language Models).

----------------
### Data Directory
Please download the model weight from the following link and put it under the "models" directory:

https://drive.google.com/file/d/1KyZ3mnnVZKcCV2ge0rFAEhP4zPIKgToe/view?usp=sharing
<pre><code>
KEEN
├──data
│   ├──test.csv
├──models
│   ├──pretrained_model.pth
├──config.py
├──train.py
      .
      .
      .
</code></pre>

### Requirements
* numpy==1.22.4
* requests==2.32.3
* scikit-learn==1.5.0
* scipy==1.13.1
* tokenizers==0.19.1
* torch==2.3.1
* torchaudio==2.3.1
* torchvision==0.18.1
* tqdm==4.66.4
* transformers==4.42.3
* x-transformers==1.31.6

### Run
To inference the code, please use the following command:
<pre><code>
python main.py --task inference
</code></pre>



