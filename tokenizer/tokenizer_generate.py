import json

# Module-level vocab cache: avoid reloading the 50K-entry JSON on every tokenizer call
_VOCAB_CACHE: dict[str, list] = {}

class BiDirectctionMatchingMethod(object):

    def __init__(self, text, vocab_file):
        self.dic = self.load_vocab(vocab_file)
        self.text = text

    def load_vocab(self, vocab_file):
        if vocab_file not in _VOCAB_CACHE:
            with open(vocab_file, 'r', encoding='utf-8') as file:
                vocab_data = json.load(file)
            # Use frozenset for O(1) membership test instead of O(n) list scan
            _VOCAB_CACHE[vocab_file] = frozenset(vocab_data.keys())
        return _VOCAB_CACHE[vocab_file]

    def MM_cut(self, max_len=26):
        MM_result = []
        MM_index = 0
        MM_text_length = len(self.text)

        while MM_index < MM_text_length:
            max_match = None
            
            # 尝试从最大长度开始匹配
            for size in range(max_len, 0, -1):  # 从最大长度开始逐步减少
                if MM_index + size <= MM_text_length:
                    MM_piece = self.text[MM_index: MM_index + size]
                    if MM_piece in self.dic:
                        max_match = MM_piece
                        MM_index += size  # 找到匹配，更新索引
                        break

            # 如果没有匹配到，逐字匹配
            if max_match is None:
                max_match = self.text[MM_index]
                MM_index += 1  # 移动一个字符

            MM_result.append(max_match)

        return MM_result

    def RMM_cut(self, max_len=26):
        RMM_result = []
        RMM_index = len(self.text)

        while RMM_index > 0:
            max_match = None
            
            # 尝试从最大长度开始匹配
            for size in range(max_len, 0, -1):  # 从最大长度开始逐步减少
                if RMM_index - size >= 0:
                    RMM_piece = self.text[RMM_index - size: RMM_index]
                    if RMM_piece in self.dic:
                        max_match = RMM_piece
                        RMM_index -= size  # 找到匹配，更新索引
                        break

            # 如果没有匹配到，逐字匹配
            if max_match is None:
                max_match = self.text[RMM_index - 1]
                RMM_index -= 1  # 移动一个字符

            RMM_result.append(max_match)

        RMM_result.reverse()  # 结果反转为从左到右顺序
        return RMM_result


def get_best_matching_result(MM_result, RMM_result):
    if len(MM_result) != len(RMM_result):
        # 如果两个结果词数不同，返回词数较少的那个
        return MM_result if (len(MM_result) < len(RMM_result)) else RMM_result
    else:
        if MM_result == RMM_result:
            # 词典结果相同，返回任意一个
            return MM_result
        else:
            # 词数相同但是词典结果不同，返回单字最少的那个
            MM_word_1 = 0
            RMM_word_1 = 0
            for word in MM_result:
                if len(word) == 1:
                    MM_word_1 += 1

            for word in RMM_result:
                if len(word) == 1:
                    RMM_word_1 += 1

            return MM_result if (MM_word_1 < RMM_word_1) else RMM_result


if __name__ == '__main__':
    # 测试
    text = 'C1CCCC1OBrClSi'
    vocab_file = "data_rl/vocab_rl/merged_vocab_rl.json"  # 这里是你的词汇文件路径
    tokenizer = BiDirectctionMatchingMethod(text, vocab_file)

    # 计算词典中最长和第二长的词条的长度并找到相应的词条
    sorted_vocab = sorted(tokenizer.dic, key=len, reverse=True)  # 按长度降序排序
    longest_word = sorted_vocab[0]  # 最长的词
    rl_longest_word = sorted_vocab[25] if len(sorted_vocab) > 25 else None  # 第二长的词

    print(f"The longest word in the vocab is: {longest_word} with length {len(longest_word)}")
    print(f"The second longest word in the vocab is: {rl_longest_word} with length {len(rl_longest_word) if rl_longest_word else 'N/A'}")

    # 使用第二长的词条长度进行分词
    max_len = len(rl_longest_word) if rl_longest_word else len(longest_word)

    MM_result = tokenizer.MM_cut(max_len)
    RMM_result = tokenizer.RMM_cut(max_len)
    best_result = get_best_matching_result(MM_result, RMM_result)

    print("Best result:", best_result)

    a = 1