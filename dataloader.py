from torch.utils.data import Dataset


class GenerateDataset(Dataset):
    def __init__(self, args, sequences, substrates, sequence_tokenizer, substrate_tokenizer, labels=None):
        super(GenerateDataset, self).__init__()
        self.args = args
        self.sequences = sequences
        self.substrates = substrates
        self.labels = labels
        self.sequence_tokenizer = sequence_tokenizer
        self.substrate_tokenizer = substrate_tokenizer

    def __getitem__(self, idx):
        # tokenize protein
        sequence_output = self.sequence_tokenizer(self.sequences[idx], add_special_tokens=True, return_tensors="pt",
                                                  padding='max_length', max_length=self.args.protein_max_len, truncation=True)

        # tokenize substrate
        substrate_output = self.substrate_tokenizer(self.substrates[idx], add_special_tokens=True, return_tensors="pt",
                                                    truncation=True, max_length=self.args.substrate_max_len,
                                                    padding='max_length', return_offsets_mapping=False)
        if self.labels is not None:
            return sequence_output, substrate_output, self.labels[idx]
        else:
            return sequence_output, substrate_output

    def __len__(self):
        return len(self.sequences)
















