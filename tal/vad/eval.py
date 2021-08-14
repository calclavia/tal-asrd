import pickle
import os
import numpy as np
from tqdm import tqdm

base_dir = '/home/shuyang/data4/vad'

with open(os.path.join(base_dir, 'bounds.pkl'), 'rb') as rf:
    bounds = pickle.load(rf)

for label, floc in [
    ('webrtc', os.path.join(base_dir, 'webrtc-sad.pickle')),
    ('pretrained-AMI', os.path.join(base_dir, 'pretrained-sad.pickle')),
]:
    with open(floc, 'rb') as rf:
        all_hyp = pickle.load(rf)
    print('\n===== {}'.format(label))
    all_ref_frames = 0
    all_ref_speech = 0
    all_ref_noise = 0
    all_hyp_speech = 0
    all_hyp_noise = 0
    all_correct = 0
    all_s_as_s = 0
    all_n_as_s = 0
    for ep in tqdm(all_hyp):
        # Hundredth-second intervals
        dur = bounds['test'][ep]['duration']
        max_len = int(dur * 100)
        ref = bounds['test'][ep]['bounds']
        hyp = all_hyp[ep]
        # Reference array
        ref_array = np.zeros(max_len).astype(int)
        for start, end in ref:
            ref_array[int(start * 100):int(end * 100)] = 1
        # Hypothesis array
        hyp_array = np.zeros(max_len).astype(int)
        for start, end in hyp:
            hyp_array[int(start * 100):int(end * 100)] = 1
        # Values
        ref_v_hyp = list(zip(ref_array, hyp_array))
        total = len(ref_v_hyp)
        all_ref_frames += total
        n_correct = (ref_array == hyp_array).sum()
        all_correct += n_correct
        n_wrong = (ref_array != hyp_array).sum()
        total_speech = (ref_array == 1).sum()
        all_ref_speech += total_speech
        total_noise = (ref_array == 0).sum()
        all_ref_noise += total_noise
        total_pred_speech = (hyp_array == 1).sum()
        all_hyp_speech += total_pred_speech
        total_pred_noise = (hyp_array == 0).sum()
        all_hyp_noise += total_pred_noise
        speech_as_speech = 0
        noise_as_noise = 0
        noise_as_speech = 0
        speech_as_noise = 0
        # Breakdown
        for r, h in ref_v_hyp:
            if r == 1 and h == 0:
                speech_as_noise += 1
            if r == 0 and h == 1:
                noise_as_speech += 1
            if r == 1 and h == 1:
                speech_as_speech += 1
            if r == 0 and h == 0:
                noise_as_noise += 1
        all_s_as_s += speech_as_speech
        all_n_as_s += noise_as_speech
    speech_precision = all_s_as_s / all_hyp_speech
    speech_recall = all_s_as_s / all_ref_speech
    speech_f1 = 2 * speech_precision * speech_recall / (speech_precision + speech_recall)
    print('Compression rate: {:.2f}% preserved'.format(100 * all_hyp_speech / all_ref_frames))
    print('Total % Speech: {:.2f}%'.format(100 * all_ref_speech / all_ref_frames))
    print('Correct frame prediction: {:.2f}%'.format(100 * all_correct / all_ref_frames))
    print('Speech precision: {:.2f}%'.format(100 * speech_precision))
    print('Speech recall: {:.2f}%'.format(100 * speech_recall))
    print('Speech F1: {:.2f}%'.format(100 * speech_f1))
    print('Noise classified as speech: {:.2f}%'.format(100 * all_n_as_s / all_ref_noise))


