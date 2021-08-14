import torch, os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights_paths', nargs='+', type=str)
    parser.add_argument('--start-epoch', type=int, help='Starting epoch of checkpoints')
    parser.add_argument('--end-epoch', type=int, help='Ending epoch of checkpoints')
    parser.add_argument('--out-path', type=str, required=True)

    args = parser.parse_args()
    
    current_state = None
    with torch.no_grad():
        weight_paths = args.weights_paths

        if len(weight_paths) == 1:
            print('Using weight path as folder prefix', weight_paths)
            weight_paths = [os.path.join(weight_paths[0], '_ckpt_epoch_{}.ckpt'.format(i)) for i in range(args.start_epoch, args.end_epoch)]

        for path in weight_paths:
            print('Loading', path)
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
            if current_state is None:
                current_state = {k1: v1 / len(args.weights_paths) for k1, v1 in checkpoint['state_dict'].items()}
            else:
                current_state = {k1: v1 + v2 / len(args.weights_paths) for (k1, v1), (k2, v2)  in zip(current_state.items(), checkpoint['state_dict'].items())}

    torch.save({'state_dict': current_state}, args.out_path)
    print('Saved.')