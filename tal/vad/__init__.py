# Requires python 3.7
"""
apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.7
apt-get install -y python3.7-dev
"""

# Install dependencies
"""
python3.7 -m pip install torch==1.4.0
python3.7 -m pip install pytorch-lightning==0.7.1
python3.7 -m pip install setuptools==46.1.3
python3.7 -m pip install pyannote.core==3.7.1
python3.7 -m pip install pyannote.audio==2.0a1
python3.7 -m pip install mutagen==1.44.0
python3.7 -m pip install librosa==0.7.2
python3.7 -m pip install tqdm==4.45.0
python3.7 -m pip install editdistance==0.5.3
python3.7 -m pip install colorama==0.4.3
"""

if __name__ == "__main__":
    import os
    import json
    import pickle
    import numpy as np
    from tqdm import tqdm
    import mutagen
    from librosa.core import get_duration

    # Setup
    base_dir = '/home/shuyang/data4/tal-data'
    vad_dir = '/home/shuyang/data4/vad'

    print('Base dir: {}'.format(base_dir))

    # Test one file
    spk_loc = os.path.join(vad_dir, 'speakers.pkl')
    bounds_loc = os.path.join(vad_dir, 'bounds.pkl')

    all_speakers = []

    all_speech_segments = {
        'train': dict(), 'valid': dict(), 'test': dict()
    }

    for split in ['train', 'valid', 'test']:
        print('Processing split: {}'.format(split))
        rttm_loc = os.path.join(vad_dir, 'TAL.{}.rttm'.format(split))
        rttm_rows = []
        uem_loc = os.path.join(vad_dir, 'TAL.{}.uem'.format(split))
        uem_rows = []
        # Dictionary of ep-XXX : conversation dict
        transcript_loc = os.path.join(base_dir, split, 'transcript.pkl')
        with open(transcript_loc, 'rb') as rf:
            transcripts = pickle.load(rf)
        ep_tqdm = tqdm(transcripts.items())
        for ep, ep_turns in ep_tqdm:
            ep_tqdm.set_description('Episode: {}'.format(ep))
            wav_loc = os.path.join(base_dir, split, '{}.wav'.format(ep))
            wav_len = get_duration(filename=wav_loc)
            # UEM
            uem_row = '{uri} 1 {start} {end}'.format(
                uri=ep,
                start='0.000',
                end='{:.6f}'.format(wav_len)
            )
            uem_rows.append(uem_row)
            all_speech_segments[split][ep] = {
                'duration': wav_len,
                'bounds': [],
            }
            for turn in ep_turns:
                u_start = turn['utterance_start']
                u_end = turn['utterance_end']
                # Invalid segment
                if u_start is None or u_end is None:
                    continue
                # End NaN -> unable to identify the segment bounds
                if np.isnan(u_end) or np.isnan(u_start):
                    continue
                # Invalid bounds
                if u_start > u_end:
                    continue
                # Duration
                duration = u_end - u_start
                # Speaker ID
                speaker_name = turn['speaker'].lower().strip()
                try:
                    speaker_id = all_speakers.index(speaker_name)
                except:
                    speaker_id = len(all_speakers)
                    all_speakers.append(speaker_name)
                # Unique ID
                speaker_uri = 'SP{}'.format(speaker_id)
                # We want this written in RTTM format
                all_speech_segments[split][ep]['bounds'].append(
                    (u_start, u_end)
                )
                rttm_row = 'SPEAKER {uri} 1 {start} {duration} <NA> <NA> {identifier} <NA> <NA>'.format(
                    uri=ep,
                    start='{:.6f}'.format(u_start),
                    duration='{:.6f}'.format(duration),
                    identifier=speaker_uri,
                )
                rttm_rows.append(rttm_row)
        # Write RTTM file
        with open(rttm_loc, 'w+') as wf:
            for r in tqdm(rttm_rows):
                _ = wf.write(r)
                _ = wf.write('\n')
        print('{} RTTM dumped to {} ({:.3f} MB on disk)'.format(
            split, rttm_loc, os.path.getsize(rttm_loc) / 1024 / 1024
        ))
        # Write UEM file
        with open(uem_loc, 'w+') as wf:
            for r in tqdm(uem_rows):
                _ = wf.write(r)
                _ = wf.write('\n')
        print('{} UEM dumped to {} ({:.3f} MB on disk)'.format(
            split, uem_loc, os.path.getsize(uem_loc) / 1024 / 1024
        ))

    # Speakers
    with open(spk_loc, 'wb') as wf:
        pickle.dump(all_speakers, wf)

    print('Speakers dumped to {} ({:.3f} MB on disk)'.format(
        spk_loc, os.path.getsize(spk_loc) / 1024 / 1024
    ))

    # Bounds
    with open(bounds_loc, 'wb') as wf:
        pickle.dump(all_speech_segments, wf)

    print('Segments dumped to {} ({:.3f} MB on disk)'.format(
        bounds_loc, os.path.getsize(bounds_loc) / 1024 / 1024
    ))

    # Clone pyannote-audio for dev mode
    """
    git clone https://github.com/pyannote/pyannote-audio.git
    cd pyannote-audio
    git checkout develop
    python3.7 -m pip install .
    export TUTORIAL_DIR=/home/shuyang/data4/pyannote-audio/tutorials/data_preparation
    export VAD_DIR=/home/shuyang/data4/vad
    # MUSAN noise database
    cd ${TUTORIAL_DIR}
    sh download_musan.sh /home/shuyang/data4/vad
    python3.7 -m pip install pyannote.db.musan
    """

    # Database setup
    """
    export VAD_DIR=/home/shuyang/data4/vad
    vim ${VAD_DIR}/database.yml
    """

    # Write the database configuration
    """
    Databases:
    TAL: /home/shuyang/data4/tal-data/*/{uri}.wav
    MUSAN: /home/shuyang/data4/vad/musan/{uri}.wav

    Protocols:
    TAL:
        SpeakerDiarization:
            episodes:
            train:
                annotation: /home/shuyang/data4/vad/TAL.train.rttm
                annotated: /home/shuyang/data4/vad/TAL.train.uem
            development:
                annotation: /home/shuyang/data4/vad/TAL.valid.rttm
                annotated: /home/shuyang/data4/vad/TAL.valid.uem
            test:
                annotation: /home/shuyang/data4/vad/TAL.test.rttm
                annotated: /home/shuyang/data4/vad/TAL.test.uem
    """

    # Export the database config:
    """
    export PYANNOTE_DATABASE_CONFIG=${VAD_DIR}/database.yml
    cat ${PYANNOTE_DATABASE_CONFIG}
    cd ${VAD_DIR}
    """

    import os
    import pickle

    # Load pipeline
    import torch
    sad = torch.hub.load('pyannote/pyannote-audio', 'sad_ami')
    print('Loaded SAD pipeline')

    # Load pipeline (pretrained)
    from pyannote.database import get_protocol
    from pyannote.database import FileFinder
    from datetime import datetime
    from librosa.core import get_duration

    preprocessors = {'audio': FileFinder()}
    protocol = get_protocol(
        'TAL.SpeakerDiarization.episodes',
        preprocessors=preprocessors
    )

    start_all = datetime.now()
    pt_sad_bounds_loc = '/home/shuyang/data4/vad/pretrained-sad.pickle'
    pretrained_test_bounds = dict()
    for test_file in list(protocol.test()):
        episode = test_file['uri']
        start = datetime.now()
        sad_scores = sad(test_file)
        print('Recorded speech activity for {} in {}'.format(
            test_file['uri'], datetime.now() - start
        ))
        # binarize raw SAD scores
        # NOTE: both onset/offset values were tuned on AMI dataset.
        # you might need to use different values for better results.
        from pyannote.audio.utils.signal import Binarize
        binarize = Binarize(offset=0.52, onset=0.52, log_scale=True, 
                            min_duration_off=0.1, min_duration_on=0.1)
        # speech regions (as `pyannote.core.Timeline` instance)
        speech = binarize.apply(sad_scores, dimension=1)
        segments = [(seg.start, seg.end) for seg in speech]
        pretrained_test_bounds[episode] = segments

    with open(pt_sad_bounds_loc, 'wb') as wf:
        pickle.dump(pretrained_test_bounds, wf)

    print('{} - Dumped speech segment bounds to {} ({:.3f} MB)'.format(
        datetime.now() - start_all,
        pt_sad_bounds_loc,
        os.path.getsize(pt_sad_bounds_loc) / 1024 / 1024,
    ))

    print('DONE in {}'.format(
        datetime.now() - start_all
    ))


    ####################################
    # FINE TUNING
    ####################################
    """
    export EXP_DIR=/home/shuyang/data4/pyannote-audio/tutorials/finetune
    export PYANNOTE_DATABASE_CONFIG=${VAD_DIR}/database.yml
    cd ${EXP_DIR}
    cp ${PYANNOTE_DATABASE_CONFIG} database.yml
    cat database.yml
    pyannote-audio sad train --pretrained=sad_ami --subset=train --to=4 --parallel=2 --gpu ${EXP_DIR} TAL.SpeakerDiarization.episodes
    """


