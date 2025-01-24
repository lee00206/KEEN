import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, EsmModel


class KcatModel(nn.Module):
    def __init__(self, args, dropout=0.1):
        super(KcatModel, self).__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # models
        self.esm_config = AutoConfig.from_pretrained(args.protein_model_checkpoint, resume_download=True)
        self.esm_model = EsmModel.from_pretrained(args.protein_model_checkpoint, cache_dir='./cache',
                                                  output_attentions=False, resume_download=True).to(
            self.device)

        self.chemberta_config = AutoConfig.from_pretrained(args.substrate_model_checkpoint, output_hidden_states=True,
                                                           resume_download=True)
        self.chemberta_model = AutoModel.from_pretrained(args.substrate_model_checkpoint, config=self.chemberta_config,
                                                         cache_dir='./cache', resume_download=True).to(self.device)

        # classification layer
        self.dim_size = self.chemberta_config.hidden_size + self.esm_config.hidden_size
        self.linear = nn.Linear(self.dim_size, args.d_model)
        self.classifier = nn.Linear(args.d_model, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.batchnorm1 = nn.BatchNorm1d(args.d_model)
        self.linear2 = nn.Linear(args.d_model, args.d_model)
        self.batchnorm2 = nn.BatchNorm1d(args.d_model)
        self.dropout2 = nn.Dropout(dropout)

    def feature(self, sequences, substrates):
        sequences['input_ids'] = sequences.input_ids.squeeze(1)
        sequences['attention_mask'] = sequences.attention_mask.squeeze(1)

        substrates['input_ids'] = substrates.input_ids.squeeze(1)
        substrates['attention_mask'] = substrates.attention_mask.squeeze(1)

        sequences['input_ids'] = sequences['input_ids'].to(self.device)
        sequences['attention_mask'] = sequences['attention_mask'].to(self.device)

        substrates['input_ids'] = substrates['input_ids'].to(self.device)
        substrates['attention_mask'] = substrates['attention_mask'].to(self.device)

        sequence_output = self.esm_model(**sequences)  # [batch_size, max_len, hidden_dim=480]
        substrate_output = self.chemberta_model(**substrates)  # [batch_size, max_len, hidden_dim=384]

        sequence_output = sequence_output.pooler_output
        substrate_output = substrate_output.pooler_output

        sequence_output = sequence_output / sequence_output.norm(dim=1, keepdim=True)
        substrate_output = substrate_output / substrate_output.norm(dim=1, keepdim=True)

        return sequence_output, substrate_output

    def forward(self, sequences, substrates):
        proteins, substrates = self.feature(sequences, substrates)

        combined_features = torch.cat((proteins, substrates), dim=-1)

        combined_features = self.linear(combined_features)
        _x = combined_features
        combined_features = self.relu(self.dropout(self.batchnorm1(combined_features)))  # 1st layer

        combined_features = self.linear2(combined_features)
        combined_features = self.relu(self.dropout2(self.batchnorm2(combined_features)))  # 2nd layer
        combined_features = combined_features + _x

        predictions = self.classifier(combined_features)
        predictions = predictions.squeeze(-1)

        return predictions





