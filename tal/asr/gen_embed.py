from tqdm import tqdm
from apex import amp
from torch.utils.data import DataLoader
from .data import ASRAlignedDataset, ASRAlignedCollater, PretrainDataset, PretrainCollator, RandomSegmentDataset, AudioCollator,  ASRSegmentDataset, ASRSegmentCollater
from wildspeech.lm.model import DecoderLMModel
from .args import get_argparser
from .system import System
from pytorch_lightning import Trainer
import torch
import os
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


"""
Generates speaker embeddings from training set.
python -m wildspeech.asr.gen_embed --train-data ./data/tal-final/train/ --valid-data ./data/tal-final/valid/ --test-data ./data/tal-final/test/ --tokenizer taltoken-cased.model --model-type 2x --load ./models/wild-speech/tal-tds-speaker-3/checkpoints/_ckpt_epoch_49.ckpt --num-speakers 6008 --batch-size 128 --out-path ./models/wild-speech/tal-tds-speaker-3/train_spk_embeds
"""
if __name__ == '__main__':
    parser = get_argparser()
    parser.add_argument('--out-path', type=str, required=True,
                        help='Output path to dump speaker embeddings')
    args = parser.parse_args()
    print(args)

    assert args.load is not None, 'Specify path to weights.'

    if not args.val_batch_size:
        args.val_batch_size = args.batch_size

    # TODO: Update to use Lightning's official loading
    # load on CPU only to avoid OOM issues
    # then its up to user to put back on GPUs
    checkpoint = torch.load(
        args.load, map_location=lambda storage, loc: storage)
    state_dict = checkpoint['state_dict']
    # TODO: A hack to remove the tokens
    # state_dict['model.token_embedding.weight'] = state_dict['model.token_embedding.weight'][:10000]
    # state_dict['model.lm_head.weight'] = state_dict['model.lm_head.weight'][:10000]

    # load the state_dict on the model automatically
    system = System(args)
    system.load_state_dict(state_dict, strict=not args.no_strict)

    # give model a chance to load something
    system.on_load_checkpoint(checkpoint)
    system = system.eval().cuda()
    # Prevents Pytorch unsupported FP16 modules from erroring
    system = amp.initialize(system, opt_level='O1')

    dataset = ASRAlignedDataset(
        args.train_data[0],
        system.tokenizer,
        num_utterances=1,
        max_segment_duration=args.max_secs,
        speaker_map_loc=os.path.join(
            args.train_data[0], 'speaker_map.json') if args.num_speakers > 0 else None
    )
    dloader = DataLoader(
        dataset,
        num_workers=system.num_workers,
        batch_size=system.val_bsz,
        collate_fn=ASRAlignedCollater(system.tokenizer.pad_token_id),
        pin_memory=True)
    print('Loaded aligned dataset', len(dataset))

    with torch.no_grad():
        all_speakers = None
        all_speaker_ids = None

        speaker_embeddings = system.model.embedding.weight[len(
            system.tokenizer):]

        for batch in tqdm(dloader):
            x, audio_lens, y, y_mask = map(lambda x: x.cuda(), batch[:4])
            x = x.half()
            y_prev = y[:, :-1]
            y_target = y[:, 1:]

            # Positions where it's supposed to be a speaker output
            speaker_mask = y_target >= len(system.tokenizer)
            speaker_pos = speaker_mask.nonzero()

            logits, _ = system(x, y_prev, audio_lens)

            speaker_ids = y_target.masked_select(
                speaker_mask) - len(system.tokenizer)

            # Extract positions where we generated speaker tokens
            logits = logits.masked_select(
                speaker_mask.unsqueeze(-1)).view(-1, logits.size(-1))

            # Convert to embedding
            speaker_probs = torch.softmax(
                logits[:, len(system.tokenizer):], dim=-1)
            cur_spk_embeds = torch.matmul(speaker_probs, speaker_embeddings)

            if all_speakers is None:
                all_speakers = cur_spk_embeds
            else:
                all_speakers = torch.cat((all_speakers, cur_spk_embeds), dim=0)

            if all_speaker_ids is None:
                all_speaker_ids = speaker_ids
            else:
                all_speaker_ids = torch.cat(
                    (all_speaker_ids, speaker_ids), dim=0)

        print('Generated embeddings', all_speakers.size(), all_speaker_ids.size())

        print('Saving...')
        # Train, val split
        all_speakers = all_speakers.cpu()
        all_speaker_ids = all_speaker_ids.cpu()

        num_train = int(len(all_speakers) * 0.8)
        torch.save(
            (all_speakers[:num_train], all_speaker_ids[:num_train]), args.out_path + '.train.pt')
        torch.save(
            (all_speakers[num_train:], all_speaker_ids[num_train:]), args.out_path + '.valid.pt')
        print('Done.')
