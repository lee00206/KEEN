import os
import numpy as np
import pandas as pd
import random
import torch
import pprint
from config import get_Config
from tester import Tester

def main(args):
    print('<---------------- Parameters ---------------->')
    pprint.pprint(args)

    # seed
    np.seterr(all="ignore")
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    df = pd.read_csv(f'{args.inference_file_path}')
    if args.task == 'inference':
        sequences, substrates = df.sequences, df.smiles
    elif args.task == 'test':
        sequences, substrates, labels = df.sequences, df.smiles, df.labels
    tester = Tester(args, sequences, substrates)
    preds = tester.inference()

    if args.output_file_path is not None:
        df['predictions'] = preds
        df.to_csv(f'{args.output_file_path}', index=False)
        print('Saved the output file!')


if __name__ == "__main__":
    args = get_Config()

    main(args)



