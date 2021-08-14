import os
from typing import Sequence
from transformers import GPT2Tokenizer

from wildspeech.asr.tokenizers import _Tokenizer
import logging

# Suppress tokenization warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

class Tokenizer(_Tokenizer):
    def __init__(self, cache_path: str = None, **kwargs):
        # Initial tokenizer
        super().__init__(cache_path, **kwargs)

        # Special kwargs
        cache_path = kwargs.get('cache_path')
        model_class = kwargs.get('tokenizer_class', GPT2Tokenizer)
        pretrained_model = kwargs.get('pretrained_model', 'gpt2')

        # Load from cache
        self.tokenizer = model_class.from_pretrained(
            pretrained_model,
            cache_dir=os.path.join(cache_path, pretrained_model) if cache_path else None
        )

        # Tokens
        self._bos_token_id = 49129 # BOS
        self._eos_token_id = self.tokenizer.eos_token_id  # EOS
        self._pad_token_id = self.tokenizer.pad_token_id or 0  # Padding
        self._eot_token_id = 49129  # EOT: end of transcript token

    def __len__(self):
        return len(self.tokenizer)

    def _encode(self, sentence: str, **kwargs) -> Sequence[int]:
        """
        Encodes a sentence to token IDs
        
        Args:
            sentence (str): Sentence in text form
        
        Returns:
            Sequence[int]: List of token IDs (integers)
        """
        return self.tokenizer.encode(sentence, **kwargs)

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
            clear_buffer = x == self.eot_token_id or x >= len(self.tokenizer)

            if clear_buffer:
                if len(buffer) > 0:
                    output_str += self.tokenizer.decode(buffer)
                buffer = []

            if x == self.eot_token_id:
                output_str += '<EOT>'
            elif x >= len(self.tokenizer):
                output_str += '<S{}>'.format(x - len(self.tokenizer))
            
            else:
                buffer.append(x)
        if len(buffer) > 0:
            output_str += self.tokenizer.decode(buffer)
        return output_str
