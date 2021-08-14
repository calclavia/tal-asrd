import json
import pickle

"""
Post processes the jsonl outputs from test
"""
with open('out/ref.jsonl') as f:
    refs = [json.loads(x.strip()) for x in f]
with open('out/hyp.jsonl') as f:
    hyps = [json.loads(x.strip()) for x in f]

zipped = list(zip(refs, hyps))
# zipped = [list(zip(ref_utts, hyp_utts)) for ref_utts, hyp_utts in zipped]
# zipped = [[(ref_utt[0], hyp_utt[0]) for ref_utt, hyp_utt in zip(ref_utts, hyp_utts)] for ref_utts, hyp_utts in zipped]

print('== Example ==')
print(zipped[0])

with open('out/test_result.pkl', 'wb') as f:
    pickle.dump(zipped, f)
print('Done')