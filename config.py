import argparse


def get_Config():
    parser = argparse.ArgumentParser()

    # protein embedding parameter
    parser.add_argument('--protein_model_checkpoint', type=str, default='facebook/esm2_t12_35M_UR50D')
    parser.add_argument('--protein_max_len', type=int, default=768)

    # substrate embedding parameter
    parser.add_argument('--substrate_model_checkpoint', type=str, default='DeepChem/ChemBERTa-77M-MLM')
    parser.add_argument('--substrate_max_len', type=int, default=512)

    # model parameters
    parser.add_argument('--d_model', type=int, default=768, help='hidden dimension')
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--n_heads', type=int, default=12)

    # inference parameters
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--task', type=str, default='inference', help='inference, test')
    parser.add_argument('--inference_file_path', type=str, default='data/test_file.csv')
    parser.add_argument('--output_file_path', type=str, default=None)

    # hardware parameters
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    return args





