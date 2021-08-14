from abc import ABCMeta, abstractmethod
from typing import Union, Sequence

import torch


class _Tokenizer(metaclass=ABCMeta):
    def __init__(self, cache_path: str = None, **kwargs):

        # Load from cache
        self.cache_path = cache_path

        # Tokens
        self._bos_token_id = 0  # BOS
        self._eos_token_id = 1  # EOS
        self._pad_token_id = 2  # Padding
        self._eot_token_id = 49129  # EOT: end of transcript token

    def __len__(self):
        return 0

    @property
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def bos_token_id(self):
        return self._bos_token_id

    @property
    def eos_token_id(self):
        return self._eos_token_id

    @property
    def eot_token_id(self):
        return self._eot_token_id

    @abstractmethod
    def _encode(self, sentence: str, **kwargs) -> Sequence[int]:
        """
        Encodes a sentence to token IDs
        
        Args:
            sentence (str): Sentence in text form
        
        Returns:
            Sequence[int]: List of token IDs (integers)
        """
        return NotImplemented

    def encode(self,
               sentence: str,
               bos_token: bool = True,
               eos_token: bool = True,
               **kwargs) -> Sequence[int]:
        """
        Encodes a sentence to token IDs
        
        Args:
            sentence (str): Sentence in text form
            bos_token (bool, optional): Whether to add a BOS token. Defaults to True.
            eos_token (bool, optional): Whether to add an EOS token. Defaults to True.
        
        Returns:
            Sequence[int]: List of token IDs (integers)
        """
        sentence_tokens = self._encode(sentence, **kwargs)
        if bos_token:
            sentence_tokens = [self.bos_token_id] + sentence_tokens
        if eos_token:
            sentence_tokens += [self.eos_token_id]

        return sentence_tokens

    @abstractmethod
    def decode_list(self, tokens: list) -> str:
        """
        Decode a list of IDs
        
        Args:
            tokens (list): List of token IDs
        
        Returns:
            str: Output strings
        """
        return NotImplementedError

    def decode(self, tokens: Union[list, torch.LongTensor]) -> str:
        """
        Decodes a sequence of IDs into a string
        
        Args:
            tokens (Union[list, torch.LongTensor]): Tokens sequence
        
        Returns:
            str: Output text
        """
        if isinstance(tokens, torch.Tensor):
            return self.decode_list(tokens.cpu().tolist())

        return self.decode_list(tokens)

    def decode_speakers(self, tokens: list, add_last=True) -> list:
        """
        Decodes a sequence of tokens. Tokens outside of the tokenizer range are treated as speaker IDs.
        
        Returns:
            [(str: Output text, speakerID), ...], List of token indices in which we made utterance splits (EOS positions)
        """
        utterances = []
        buffer = []
        split_indices = []
        cur_speaker = None

        for i, x in enumerate(tokens):
            # Ignore BOS
            if x == self.bos_token_id:
                continue

            if x >= len(self):
                cur_speaker = x - len(self)
            elif x == self.eos_token_id:
                if len(buffer) > 0:
                    # End of sentence. Break the buffer.
                    utterances.append([self.decode(buffer), cur_speaker])
                    cur_speaker = None
                    buffer = []
                    split_indices.append(i)
            else:
                buffer.append(x)

        # Last buffer
        if len(buffer) > 0 and add_last:
            utterances.append([self.decode(buffer), cur_speaker])
            split_indices.append(i)

        assert len(utterances) == len(split_indices)
        return [tuple(t) for t in utterances], split_indices
