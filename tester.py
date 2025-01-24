import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import EsmTokenizer, AutoTokenizer
from dataloader import GenerateDataset
import gc
from tqdm import tqdm
import math
from model import KcatModel
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

gc.collect()
torch.cuda.empty_cache()


class Tester(nn.Module):
    def __init__(self, args, sequences, substrates, labels=None):
        super(Tester, self).__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # tokenizer and model
        sequence_tokenizer = EsmTokenizer.from_pretrained(args.protein_model_checkpoint)
        substrate_tokenizer = AutoTokenizer.from_pretrained(args.substrate_model_checkpoint)
        self.model = KcatModel(args).to(self.device)

        self.model.load_state_dict(torch.load(f'models/pretrained_model.pth'), strict=False)
        print('Pretrained model loaded')

        # load data
        test_dataset = GenerateDataset(args, sequences, substrates, sequence_tokenizer, substrate_tokenizer, labels)
        self.test_iter = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)

    def test(self):
        test_true, test_preds = [], []

        with torch.no_grad():
            self.model.eval()
            for sequences, substrates, labels in tqdm(self.test_iter):
                sequences, substrates, labels = sequences.to(self.device), substrates.to(self.device), labels.to(self.device)
                preds = self.model(sequences, substrates)

                test_true += labels.detach().cpu().numpy().tolist()
                test_preds += preds.detach().cpu().numpy().tolist()

        # metric
        r2 = r2_score(test_true, test_preds)
        mae = mean_absolute_error(test_true, test_preds)
        pearson_r = stats.pearsonr(test_true, test_preds)

        print(f'Test R2: {r2:.4f} | MAE: {mae:.4f} | Pearson R: {pearson_r.statistic:.4f}')

        test_preds = [math.pow(10, pred) for pred in test_preds]

        return test_preds

    def inference(self):
        preds_list = []

        with torch.no_grad():
            self.model.eval()
            for sequences, substrates in tqdm(self.test_iter):
                sequences, substrates = sequences.to(self.device), substrates.to(self.device)
                preds = self.model(sequences, substrates)

                preds_list += preds.detach().cpu().numpy().tolist()

        preds = [math.pow(10, pred) for pred in preds_list]

        return preds



