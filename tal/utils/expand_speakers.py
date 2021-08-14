import os
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Expands the GPT2 tokenizer embedding of a saved model with speaker IDs.')
    parser.add_argument('in_file', type=str)
    parser.add_argument('out_file', type=str)

    args = parser.parse_args()
    print(args)

    checkpoint = torch.load(
        args.in_file, map_location=lambda storage, loc: storage)
    current_state = checkpoint['state_dict']

    with torch.no_grad():
        # Expand the embedding with speaker IDs
        embedding = current_state['model.token_embedding.weight']
        # Randomly initialize new embeddings from Gaussian distribution
        new_add = torch.randn((10000, embedding.size(1))) * 0.02
        current_state['model.token_embedding.weight'] = torch.cat((embedding, new_add), dim=0)
        current_state['model.lm_head.weight'] = torch.cat((embedding, new_add), dim=0)
        print('New embedding size', current_state['model.token_embedding.weight'].size())

    torch.save({'state_dict': current_state}, args.out_file)
