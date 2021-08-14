from typing import Sequence
from sentencepiece import SentencePieceProcessor

from wildspeech.asr.tokenizers import _Tokenizer

"""
Training the tokenizer

from sentencepiece import SentencePieceTrainer

SentencePieceTrainer.Train(
    '--input=librispeech-vocab.txt --model_prefix=libritoken --vocab_size=10000 '
    '--bos_id=0 --eos_id=1 --pad_id=2 --unk_id=3 --character_coverage=1.0 --model_type=bpe'
)
"""

class Tokenizer(_Tokenizer):
    def __init__(self, cache_path: str = None, **kwargs):
        # Initial tokenizer
        super().__init__(cache_path, **kwargs)

        # Load from cache
        self.tokenizer = SentencePieceProcessor()
        self.tokenizer.load(cache_path)

        # Tokens
        self._bos_token_id = self.tokenizer.bos_id()  # BOS
        self._eos_token_id = self.tokenizer.eos_id()  # EOS
        self._pad_token_id = self.tokenizer.pad_id()  # Padding
        self._eot_token_id = 0  # EOT: end of transcript token

    def __len__(self):
        return len(self.tokenizer)

    def __getstate__(self):
        """ Pickle serialization """
        state = self.__dict__.copy()
        del state['tokenizer']
        return state

    def __setstate__(self, state):
        """ Pickle deserialization """
        self.__dict__ = state
        self.tokenizer = SentencePieceProcessor()
        self.tokenizer.load(self.cache_path)
        
    def _encode(self, sentence: str, **kwargs) -> Sequence[int]:
        """
        Encodes a sentence to token IDs

        Args:
            sentence (str): Sentence in text form

        Returns:
            Sequence[int]: List of token IDs (integers)
        """
        return self.tokenizer.EncodeAsIds(sentence, **kwargs)

    def decode_list(self, tokens: list) -> str:
        """
        Decode a list of IDs

        Args:
            tokens (list): List of token IDs

        Returns:
            str: Output strings
        """
        output_str = ''
        buffer = []
        for x in tokens:
            clear_buffer = x == self.eot_token_id or x >= len(self)

            if clear_buffer:
                if len(buffer) > 0:
                    output_str += self.tokenizer.DecodeIds(buffer)
                buffer = []

            if x == self.eot_token_id:
                output_str += '<EOT>'
            elif x >= len(self):
                output_str += '<S{}>'.format(x - len(self))

            else:
                buffer.append(x)
        if len(buffer) > 0:
            output_str += self.tokenizer.DecodeIds(buffer)
        return output_str
