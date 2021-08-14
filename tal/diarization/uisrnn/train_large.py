"""
Train a speaker diarization model, with large data

Recommendations from:
https://github.com/google/uis-rnn#training-on-large-datasets

    Successively call `.fit(...)` on small enough sections of data
    Input sizes should be roughly equal across `.fit(...)` calls
"""
import os
import torch
import numpy as np
import pickle
import glob

from tqdm import tqdm
from datetime import datetime
from itertools import zip_longest, chain
from wildspeech import set_seed, get_device, count_parameters
from wildspeech.diarization import uisrnn

def run_experiment(
    train_f, train_cluster_f, test_f, test_cluster_f,
    model_args, training_args, inference_args, exp_name
):
    """
    Run a single experiment

    Args:
        train_f (list): Sequence of files, each containing a T x H array (audio features)
        train_cluster_f (list): Sequence of files, each containing a T x 1 array (speaker labels)
        test_f (list): Sequence of files, each containing a T x H array (audio features)
        test_cluster_f (list): Sequence of files, each containing a T x 1 array (speaker labels)
        model_args (args): Model arguments (see `arguments.py`)
        training_args (args): Training arguments (see `arguments.py`)
        inference_args (args): Inference arguments (see `arguments.py`)
        exp_name (str): Experiment name

    Returns:
        float: Accuracy over the test set
    """
    start = datetime.now()

    if training_args.debug:
        print('\n\n===== DEBUG MODE =====\n\n')

    def debug(m):
        if training_args.debug:
            print('DEBUG: {}'.format(datetime.now() - start))
            print(m)

    # Create model class
    model = uisrnn.UISRNN(model_args)
    print('{} - Created {} model with {:,} params:'.format(
        datetime.now() - start, model.__class__.__name__,
        count_parameters(model.rnn_model)
    ))
    print(model.rnn_model)

    # Training
    os.makedirs(training_args.out_dir, exist_ok=True)
    model_loc = os.path.join(training_args.out_dir, exp_name)
    model_constructed = (not training_args.overwrite) \
         and os.path.exists(model_loc)
    if model_constructed:
        try:
            model.load(model_loc)
            print('{} - Loaded trained model from {}'.format(
                datetime.now() - start, model_loc,
            ))
        except Exception as e:
            print('Unable to load model from {}:\n{}'.format(
                model_loc, e
            ))
            model_constructed = False
    if not model_constructed:
        train_f_ixs = [
            list(filter(None.__ne__, f_ix)) for f_ix in
            zip_longest(
                *[iter(range(len(train_f)))] * 10,  # Sequences of length 10
                fillvalue=None
            )
        ]
        for ii, train_f_ix in enumerate(train_f_ixs):
            print('Training on batch {} - {:,} files'.format(
                ii, len(train_f_ix)
            ))

            # Get sequences
            train_sequence = [
                # Sequence must be in np.float (python float / np.float64)
                np.load(train_f[i], allow_pickle=True).astype(np.float) for i in train_f_ix
            ]
            debug('Train sequence ({}): {:,} elements, first element shape {}'.format(
                type(train_sequence), len(train_sequence), train_sequence[0].shape
            ))
            train_cluster_id = [
                np.load(train_cluster_f[i], allow_pickle=True) for i in train_f_ix
            ]
            debug('Train clusters ({}): {:,} elements, first element shape {}'.format(
                type(train_cluster_id), len(train_cluster_id), train_cluster_id[0].shape
            ))

            model.fit(train_sequence, train_cluster_id, training_args)
            print('{} - Trained model!'.format(datetime.now() - start))
            model.save(model_loc)
            print('{} - CHECKPOINT - saved model to {}'.format(
                datetime.now() - start, model_loc
            ))

    # Testing
    predicted_cluster_ids = []
    test_record = []
    with torch.no_grad():
        for i, (t_f, t_c_f) in tqdm(enumerate(zip(
            test_f, test_cluster_f
        )), total=len(test_f)):
            # Load data
            # Sequence must be in np.float (python float / np.float64)
            test_seq = np.load(t_f, allow_pickle=True).astype(np.float)
            test_cluster = np.load(t_c_f, allow_pickle=True)
            debug('Test seq ({}) shape: {}'.format(
                test_seq.__class__.__name__, test_seq.shape
            ))
            debug('Test cluster ({}): {}'.format(
                test_cluster.__class__.__name__, test_cluster
            ))

            # Predict cluster centers
            predicted_cluster_id = model.predict(
                test_seq, inference_args, debug=training_args.debug
            )
            debug('Predicted cluster ID: {}, class {}'.format(
                predicted_cluster_id,
                predicted_cluster_id.__class__.__name__
            ))
            predicted_cluster_ids.append(predicted_cluster_id)
            accuracy = uisrnn.compute_sequence_match_accuracy(
                test_cluster.tolist(), predicted_cluster_id)

            # We are getting accuracy per batch
            test_record.append((accuracy, len(test_cluster)))
            debug('Gold labels: {}'.format(list(test_cluster)))
            debug('Pred labels: {}'.format(list(predicted_cluster_id)))
            debug('-' * 80)

    # Output
    output_string = uisrnn.output_result(
        model_args,
        training_args,
        test_record
    )
    print('Finished diarization experiment')
    print(output_string)
    with open(os.path.join(
        training_args.out_dir,
        '{}_test.pkl'.format(exp_name)
    ), 'wb') as wf:
        pickle.dump(test_record, wf)

    accuracy_array, _ = zip(*test_record)
    exp_accuracy = np.mean(accuracy_array)
    return exp_accuracy

def diarization_experiment(
    model_args, training_args, inference_args, data_args
):
    """Experiment pipeline.

    Load data --> train model --> test model --> output result

    Args:
    model_args: model configurations
    training_args: training configurations
    inference_args: inference configurations
    data_args: data configurations
    """
    start = datetime.now()

    # Reproducibility
    set_seed(training_args.seed)

    # k-fold cross validation
    if not training_args.cross_validation:
        raise NotImplementedError(
            'Currently, this module only supports cross-validation for large datasets!'
        )

    # Experiment
    exp_name = data_args.exp_name

    # Get sequence filenames and corresopnding clusters
    seq_files = glob.glob(data_args.train_seq)
    cluster_files = [f.replace('_seq.npy', '_cluster_id.npy') for f in seq_files]

    # Quick-test with 20 values
    if training_args.quick_test:
        print('===== QUICK TEST MODE =====')
        seq_files = seq_files[:20]
        cluster_files = cluster_files[:20]
        exp_name += '-QT'

    # Create folds
    n_folds = training_args.cross_validation
    n_files = len(seq_files)
    indices = np.arange(n_files)
    np.random.shuffle(indices)

    # Shuffle sequences and associated clusters
    seq_files = [seq_files[i] for i in indices]
    cluster_files = [cluster_files[i] for i in indices]

    # Indices for each fold
    fold_ix = [
        list(filter(None.__ne__, f_ix)) for f_ix in
        zip_longest(
            *[iter(indices)] * int(np.ceil(n_files / n_folds)),
            fillvalue=None
        )
    ]
    print(len(fold_ix))
    fold_accuracies = dict()
    for fold_n in range(n_folds):
        # Get training/testing data
        fold_exp = '{}-{}'.format(exp_name, fold_n)
        try:
            train_f = list(chain.from_iterable(
                [fold_ix[x] for x in range(n_folds) if x != fold_n]
            ))
        except:
            print('fold_ix:\n{}'.format(fold_ix))
            print('fold_n:\n{}'.format(fold_n))
            print('n_folds:\n{}'.format(n_folds))
            raise
        test_f = fold_ix[fold_n]

        def slice_seq(seq, indices):
            return [seq[i] for i in indices]

        # Run experiment
        exp_accuracy = run_experiment(
            train_f=slice_seq(seq_files, train_f),
            train_cluster_f=slice_seq(cluster_files, train_f),
            test_f=slice_seq(seq_files, test_f),
            test_cluster_f=slice_seq(cluster_files, test_f),
            model_args=model_args,
            training_args=training_args,
            inference_args=inference_args,
            exp_name=fold_exp
        )
        fold_accuracies[fold_exp] = exp_accuracy
        print('\n------------------------------------')
        print('[{}] - COMPLETE_FOLD {}: {:,.2f}% Acc ({:,.2f}% DER)'.format(
            datetime.now() - start, fold_exp, exp_accuracy * 100,
            (1 - exp_accuracy) * 100
        ))

    # Overall
    print('\n================ OVERALL =================')
    for fold_exp, acc in fold_accuracies.items():
        print('{}:\t{:,.2f}% ACC\t{:,.2f}% DER'.format(
            fold_exp, 100 * acc, 100 * (1.0 - acc)
        ))

    # Total accuracy
    total_acc = np.mean(list(fold_accuracies.values()))
    print('{:.3f}% Total accuracy'.format(total_acc * 100))
    print('{:.3f}% Total DER'.format((1.0 - total_acc) * 100))

def main():
    """The main function."""
    # Retrieve arguments
    model_args, training_args, \
        inference_args, data_args = uisrnn.parse_arguments()

    # Run experiment
    diarization_experiment(
        model_args,
        training_args,
        inference_args,
        data_args
    )

"""
==== TAL ====
python3 -u -m bernard.diarization.uisrnn.train_large --enable-cuda --batch_size 20 \
--out-dir /data4/shuyang/TAL_spk \
--train-seq "/data4/shuyang/tal-features/*_seq.npy" \
-x 5 --quick-test --debug --exp-name tal-cv

rm -rf /data4/shuyang/TAL_spk/tal*QT*
nohup python3 -u -m bernard.diarization.uisrnn.train_large --enable-cuda --batch_size 3 \
--out-dir /data4/shuyang/TAL_spk \
--train-seq "/data4/shuyang/tal-features/*_seq.npy" \
-x 5 --log-iter 1 --exp-name tal-cv > /data4/shuyang/tal-cv.log &

tail -f /data4/shuyang/tal-cv.log


rm -rf /root/data/TAL_spk/tal*QT*
nohup python3 -u -m bernard.diarization.uisrnn.train_large --enable-cuda --batch_size 3 \
--out-dir /root/data/TAL_spk \
--train-seq "/root/data/tal-features-400/*_seq.npy" \
--observation_dim 512 \
-x 5 --log-iter 1 --debug --quick-test --exp-name tal-cv > /root/data/tal-cv.log &

tail -f /root/data/tal-cv.log

nohup python3 -u -m bernard.diarization.uisrnn.train_large --enable-cuda --batch_size 50 \
--out-dir /root/data/TAL_spk \
--train-seq "/root/data/tal-features-400/*_seq.npy" \
--observation_dim 512 \
-x 5 --log-iter 20 --exp-name tal-cv > /root/data/tal-cv.log &

tail -f /root/data/tal-cv.log
"""
if __name__ == '__main__':
    main()
