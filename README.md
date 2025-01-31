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
* numpy==2.1.0
* scikit-learn==1.5.1
* torch==2.4.0
* tqdm==4.66.5
* transformers==4.44.2

### Run
To run the code, you need a csv file with 'sequences' (Protein sequences, required), 'smiles' (SMILES strings of substances, required), and 'predictions' (True kcat values, optional) columns.<br>
Please refer to the 'test_file.csv' provided to see the example format of the file.<br>

To inference (without true labels) the code, please use the following command:
<pre><code>
python main.py --task inference
</code></pre>

To test (with true labels) the code and save the results, please use the following command:
<pre><code>
python main.py --task test --output_file_path {your output file path}
</code></pre>


