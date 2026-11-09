import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import regex as re

from transformers import AddedToken, PreTrainedTokenizer
import logging
from transformers import RobertaTokenizer
from transformers.tokenization_utils_base import AddedToken

from .tokenizer_generate import BiDirectctionMatchingMethod, get_best_matching_result

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "data_rl/vocab_rl/merged_vocab_rl.json",
    "merges_file": "data_rl/merges_rl.txt",
}

@lru_cache()
def bytes_to_unicode():

    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

class AddedToken:
    def __init__(self, token: str, rstrip=False, lstrip=False, single_word=False, normalized=True):
        self.token = token
        self.rstrip = rstrip
        self.lstrip = lstrip
        self.single_word = single_word
        self.normalized = normalized

    def __repr__(self):
        return (f"AddedToken({self.token!r}, rstrip={self.rstrip}, lstrip={self.lstrip}, "
                f"single_word={self.single_word}, normalized={self.normalized})")


from transformers import PreTrainedTokenizer
import json

class PolymerSumTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = {}
    max_model_input_sizes = 512
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_list: Optional[List[str]] = None,
        vocab_file: str = "data_rl/vocab_rl/merged_vocab_rl.json",        
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        model_max_len=512,
        pretrained_path: Optional[str] = None,  # Add pretrained_path here
        **kwargs
    ):

        # 初始化时保存传入的路径
        self.pretrained_path = pretrained_path  # Save the path here

        if vocab_list is None:
            with open(vocab_file, 'r', encoding='utf-8') as file:
                vocab_data = json.load(file)
            vocab_list = list(vocab_data.keys())

        self.vocab_file = vocab_file
        self.vocab_list = vocab_list
        self.errors = errors
        self.add_prefix_space = add_prefix_space
        self.kwargs = kwargs

        # encoder must exist before super().__init__() because transformers calls
        # get_vocab() → self.encoder during _add_tokens inside __init__
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            model_max_length=model_max_len,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self.cls_token_id = self.encoder[self.cls_token]
        self.sep_token_id = self.encoder[self.sep_token]
        self.pad_token_id = self.encoder[self.pad_token]
        self.bos_token_id = self.encoder[self.bos_token]
        self.eos_token_id = self.encoder[self.eos_token]
        self.unk_token_id = self.encoder[self.unk_token]
        self.mask_token_id = self.encoder[self.mask_token]

        # 处理错误和 Byte 到 Unicode 转换
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()  # 如果需要，保留此部分
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}  # 如果需要，保留此部分

        # 初始化其他属性
        self.cache = {}
        self.add_prefix_space = add_prefix_space

    def get_vocab(self):
        return self.encoder

    def get_special_tokens_map(self):
        return {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "sep_token": self.sep_token,
            "cls_token": self.cls_token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "mask_token": self.mask_token,
        }



    def load_vocab(self, vocab_file: str) -> List[str]:
        """
        Load vocabulary from a JSON file.
        :param vocab_file: Path to the vocab file
        :return: Vocabulary as a list
        """
        with open(vocab_file, 'r', encoding='utf-8') as file:
            vocab_data = json.load(file)
        
        vocab = list(vocab_data.keys())  # Extract all the keys
        return vocab

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)
    
    def MM_cut(self) -> List[str]:
        """
        Forward Maximum Matching (FMM) segmentation
        """
        from tokenizer_generate import BiDirectionalTokenizer
        tokenizer = BiDirectionalTokenizer(self.dic, self.window_size)
        return tokenizer.MM_cut(self.text)

    def RMM_cut(self) -> List[str]:
        """
        Reverse Maximum Matching (RMM) segmentation
        """
        from tokenizer_generate import BiDirectionalTokenizer
        tokenizer = BiDirectionalTokenizer(self.dic, self.window_size)
        return tokenizer.RMM_cut(self.text)
    
    def get_best_matching_result(self, MM_result: List[str], RMM_result: List[str]) -> List[str]:
        """
        Compare and choose the best result from MM and RMM
        """
        from tokenizer_generate import get_best_matching_result
        return get_best_matching_result(MM_result, RMM_result)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text using BiMM
        :param text: Input string
        :return: Final tokenized result
        """

        bi_matcher = BiDirectctionMatchingMethod(text, self.vocab_file)

        MM_result = bi_matcher.MM_cut()    # 不需要再传 text
        RMM_result = bi_matcher.RMM_cut()

        # 比较两个分词结果
        best_result = get_best_matching_result(MM_result, RMM_result)
        
        return best_result

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.encoder.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None):
        vocab_file = f"{save_directory}/polymer_vocab.json" if filename_prefix is None else f"{save_directory}/{filename_prefix}_vocab.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, indent=2)
        return (vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)

    @classmethod
    def from_pretrained(cls, name_or_path, model_max_len, **kwargs):
        vocab_file = f"{name_or_path}/merged_vocab_rl.json"
        return cls(pretrained_path=name_or_path, vocab_file=vocab_file, model_max_len=model_max_len, **kwargs)


if __name__ == "__main__":

    vocab_file = "data_rl/vocab_rl/merged_vocab_rl.json"  # 这里是你的词汇文件路径

    with open(vocab_file, 'r', encoding='utf-8') as file:
        vocab_data = json.load(file)

    # 提取词汇表的所有键，并返回
    vocab_list = list(vocab_data.keys())

    tokenizer = PolymerSumTokenizer(pretrained_path="model_rl", max_len=512)

    # text = "C1CCCC1O"
    text = "C(=O)N^c1ccccc1^^S=O|S=O^^c1ccccc1^NC(=O)^c1ccccc1^C(=O)NC(=O)^c1ccccc1^^S=O|S=O^^c1ccccc1^C(=O)NC(=O)^c1ccccc1^"
    # text = "This is a love song"
    # text = "今天是个好日子"
    tokens = tokenizer.tokenize(text)
    print(tokens)  # ['C1CCCC1', 'O']
