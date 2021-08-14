"""
For use with: http://projector.tensorflow.org/

# tal-tds-speaker-3
python3 -u -m wildspeech.tsne_file_format --in-file /home/shuyang/data4/tal-new/synced_ordered_test.pkl --out-stub /home/shuyang/data4/tal-new/synced_ordered_test

# tal-tds-speaker-3 w/ metric @ ckpt 3
python3 -u -m wildspeech.tsne_file_format --in-file /home/shuyang/data4/tal-new/synced_ordered.pkl --out-stub /home/shuyang/data4/tal-new/synced_ordered

# tal-tds-speaker-3 w/ metric @ ckpt 13
python3 -u -m wildspeech.tsne_file_format --in-file /home/shuyang/data4/tal-new/synced-metric_ordered.pkl --out-stub /home/shuyang/data4/tal-new/synced-metric_ordered

# tal-tds-speaker-3 w/ metric @ ckpt 3
python3 -u -m wildspeech.tsne_file_format --in-file /home/shuyang/data4/tal-new/synced-metric3_ordered.pkl --out-stub /home/shuyang/data4/tal-new/synced-metric3_ordered
"""
if __name__ == "__main__":
    import os
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str, required=True)
    parser.add_argument('--out-stub', type=str, required=True)
    args = parser.parse_args()

    emb_loc = '{}-emb.tsv'.format(args.out_stub)
    label_loc = '{}-labels.tsv'.format(args.out_stub)
    role_loc = '{}-roles.tsv'.format(args.out_stub)

    episode_ex = pd.read_pickle(args.in_file)

    all_labels = []
    all_embeddings = []
    all_roles = []
    # Iterate over episodes
    for e_ref, e_hyp in episode_ex:
        # Get corresponding reference IDs (names) and hypothesis embeddings
        _, ref_id, ref_role = zip(*e_ref)
        _, hyp_tups, hyp_role = zip(*e_hyp)
        hyp_emb, _ = zip(*hyp_tups)

        # strip out Nones
        for rid, he, rr in zip(ref_id, hyp_emb, ref_role):
            if he is None:
                continue
            all_labels.append(rid)
            all_embeddings.append(he)
            all_roles.append(rr)
    
    print('{:,} total embeddings over {} episodes'.format(
        len(all_embeddings), len(episode_ex)
    ))

    # Write embedding TSV
    with open(emb_loc, 'w+') as wf:
        for emb in all_embeddings:
            _ = wf.write('\t'.join(list(map(str, emb))))
            _ = wf.write('\n')
    print('Wrote embedding TSV at {} ({:.3f} MB)'.format(
        emb_loc, os.path.getsize(emb_loc) / 1024 / 1024
    ))

    # Write 
    with open(label_loc, 'w+') as wf:
        for lab in all_labels:
            _ = wf.write(lab)
            _ = wf.write('\n')
    print('Wrote labels TSV at {} ({:.3f} MB)'.format(
        label_loc, os.path.getsize(label_loc) / 1024 / 1024
    ))

    # Write 
    with open(role_loc, 'w+') as wf:
        for ro in all_roles:
            _ = wf.write(ro)
            _ = wf.write('\n')
    print('Wrote roles TSV at {} ({:.3f} MB)'.format(
        role_loc, os.path.getsize(role_loc) / 1024 / 1024
    ))
