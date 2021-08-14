"""
Train a speaker diarization model
"""
import os
import torch
import numpy as np
import random
import pickle

from tqdm import tqdm
from datetime import datetime
from itertools import zip_longest, chain
from wildspeech import set_seed, get_device, count_parameters
from wildspeech.diarization import uisrnn

def run_experiment(
    train_sequence, train_cluster_id, test_sequence, test_cluster_id,
    model_args, training_args, inference_args, exp_name
):
    """
    Run a single experiment

    Args:
        train_sequence (np.array): Sequence of T x H arrays (audio features)
        train_cluster_id (np.array): Sequence of T x 1 arrays (speaker labels)
        test_sequence (np.array): Sequence of T x H arrays (audio features)
        test_cluster_id (np.array): Sequence of T x 1 arrays (speaker labels)
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
            print(m)

    # Create model class
    model = uisrnn.UISRNN(model_args)
    print('{} - Created {} model with {:,} params:'.format(
        datetime.now() - start, model.__class__.__name__,
        count_parameters(model.rnn_model)
    ))
    print(model.rnn_model)

    # Training
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
        model.fit(train_sequence, train_cluster_id, training_args)
        print('{} - Trained model!'.format(datetime.now() - start))
        model.save(model_loc)
        print('{} - Saved model to {}'.format(
            datetime.now() - start, model_loc
        ))

    # Testing
    predicted_cluster_ids = []
    test_record = []
    with torch.no_grad():
        for i, (test_seq, test_cluster) in tqdm(enumerate(zip(
            test_sequence, test_cluster_id
        )), total=len(test_cluster_id)):
            debug('Test seq ({}) shape: {}'.format(
                test_seq.__class__.__name__, test_seq.shape
            ))
            debug('Test cluster ({}): {}'.format(
                test_cluster.__class__.__name__, test_cluster
            ))
            predicted_cluster_id = model.predict(test_seq, inference_args)
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

    # Standard - we have a train and test split
    if not training_args.cross_validation:
        # Experiment
        exp_name = data_args.exp_name

        # Load data
        train_sequence = list(np.load(
            data_args.train_seq, allow_pickle=True
        ))
        train_cluster_id = list(np.load(
            data_args.train_clusters, allow_pickle=True
        ))
        test_sequence = list(np.load(
            data_args.test_seq, allow_pickle=True
        ))
        test_cluster_id = list(np.load(
            data_args.test_clusters, allow_pickle=True
        ))

        # Quick-test with 10 values
        if training_args.quick_test:
            print('===== QUICK TEST MODE =====')
            train_sequence = train_sequence[:10]
            train_cluster_id = train_cluster_id[:10]
            test_sequence = test_sequence[:10]
            test_cluster_id = test_cluster_id[:10]
            exp_name += '-QT'

        print('{} - Loaded data'.format(datetime.now() - start))

        # Run experiment
        exp_accuracy = run_experiment(
            train_sequence=train_sequence,
            train_cluster_id=train_cluster_id,
            test_sequence=test_sequence,
            test_cluster_id=test_cluster_id,
            model_args=model_args,
            training_args=training_args,
            inference_args=inference_args,
            exp_name=exp_name
        )
        print('\n------------------------------------')
        print('[{}] - COMPLETED {}: {:,.2f}% Acc ({:,.2f}% DER)'.format(
            datetime.now() - start, exp_name, exp_accuracy * 100,
            (1 - exp_accuracy) * 100
        ))
    # k-fold cross validation
    else:
        # Experiment
        exp_name = data_args.exp_name

        # Load data - we pass the ALL dvectors into train-X args
        all_sequence = list(np.load(
            data_args.train_seq, allow_pickle=True
        ))
        all_cluster_id = list(np.load(
            data_args.train_clusters, allow_pickle=True
        ))

        # Quick-test with 20 values
        if training_args.quick_test:
            print('===== QUICK TEST MODE =====')
            all_sequence = all_sequence[:20]
            all_cluster_id = all_cluster_id[:20]
            exp_name += '-QT'

        print('{} - Loaded data'.format(datetime.now() - start))

        # Create folds
        n_folds = training_args.cross_validation
        indices = np.arange(len(all_sequence))
        np.random.shuffle(indices)

        # Shuffle sequences and associated clusters
        all_sequence = [all_sequence[i] for i in indices]
        all_cluster_id = [all_cluster_id[i] for i in indices]

        # Indices for each fold
        fold_ix = [
            list(filter(None.__ne__, f_ix)) for f_ix in
            zip_longest(
                *[iter(indices)] * int(np.ceil(len(all_sequence) / n_folds)),
                fillvalue=None
            )
        ]
        fold_accuracies = dict()
        for fold_n in range(n_folds):
            # Get training/testing data
            fold_exp = '{}-{}'.format(exp_name, fold_n)
            train_ix = list(chain.from_iterable(
                [fold_ix[x] for x in range(n_folds) if x != fold_n]
            ))
            test_ix = fold_ix[fold_n]

            def slice_seq(seq, indices):
                return [seq[i] for i in indices]

            # Run experiment
            exp_accuracy = run_experiment(
                train_sequence=slice_seq(all_sequence, train_ix),
                train_cluster_id=slice_seq(all_cluster_id, train_ix),
                test_sequence=slice_seq(all_sequence, test_ix),
                test_cluster_id=slice_seq(all_cluster_id, test_ix),
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
==== TIMIT ====
rm -rf /data4/shuyang/TIMIT_spk/timit*QT*
nohup python3 -u -m bernard.diarization.uisrnn.train --enable-cuda --batch_size 50 \
--out-dir /data4/shuyang/TIMIT_spk \
--train-seq /data4/shuyang/TIMIT_spk/ALL_sequence.npy \
--train-clusters /data4/shuyang/TIMIT_spk/ALL_cluster_id.npy \
-x 5 --exp-name timit-cv > /data4/shuyang/timit-cv.log &

tail -f /data4/shuyang/timit-cv.log

rm -rf /root/data/temp_diarization
nohup python3 -u -m bernard.diarization.uisrnn.train --enable-cuda --batch_size 50 \
--out-dir /root/data/temp_diarization \
--train-seq /root/data/tal-features/ep-120_seq.npy \
--train-clusters /root/data/tal-features/ep-120_cluster_id.npy \
-x 5 --exp-name tal-cv-TEMP > /root/data/timit-cv-TEMP.log &

tail -f /root/data/timit-cv-TEMP.log

"""
if __name__ == '__main__':
    main()
