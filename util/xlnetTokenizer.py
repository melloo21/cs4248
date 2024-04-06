import glob
import math
import pandas as pd
import numpy as np

class XlnetTokenize:
  """ This class is to tokenize input dataset """
  @staticmethod
  def tokenize(base_tokenizer, max_token_length, sentences:list):
    full_input_ids = []
    full_input_masks = []
    full_segment_ids = []
    trimmed_sentences_idx = list()

    max_len = max_token_length

    SEG_ID_A   = 0
    SEG_ID_B   = 1
    SEG_ID_CLS = 2
    SEG_ID_SEP = 3
    SEG_ID_PAD = 4

    # Unkown Encoding
    UNK_ID = base_tokenizer.encode("<unk>")[0]
    CLS_ID = base_tokenizer.encode("<cls>")[0]
    # Sep Encoding
    SEP_ID = base_tokenizer.encode("<sep>")[0]
    MASK_ID = base_tokenizer.encode("<mask>")[0]
    EOD_ID = base_tokenizer.encode("<eod>")[0]

    for i,sentence in enumerate(sentences):
        # Tokenize sentence to token id list [Using the default transformer version -- Sentencepiece model]
        tokens_a = base_tokenizer.encode(sentence)

        # Trim the len of text []
        # print(f"sentence lenght = {len(tokens_a)}")
        if(len(tokens_a)>max_len-2):
            tokens_a = tokens_a[:max_len-2]
            trimmed_sentences_idx.append((i,len(tokens_a) ))

        tokens = []
        segment_ids = []

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(SEG_ID_A)
        # print("1 segment_ids length :: ", len(segment_ids))

        # Add <sep> token
        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_A)
        # Add <cls> token
        tokens.append(CLS_ID)
        # print("2 tokens :: ", tokens)
        segment_ids.append(SEG_ID_CLS)
        # print("3 segment_ids:: ", segment_ids)

        # Inputs
        input_ids = tokens
        # print("4 input_ids:: ", input_ids)
        # The mask has 0 for real tokens and 1 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [0] * len(input_ids)
        # print("5 input_mask :: ", input_mask)

        # Zero-pad up to the sequence length at front
        if len(input_ids) < max_len:
            # This is to ensure inputs are of the correct dimension
            delta_len = max_len - len(input_ids)
            input_ids = [0] * delta_len + input_ids
            input_mask = [1] * delta_len + input_mask
            segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        assert len(segment_ids) == max_len

        full_input_ids.append(input_ids)
        full_input_masks.append(input_mask)
        full_segment_ids.append(segment_ids)

        # Every 1000 idx print
        if i%1000==0:
            print("No.:%d"%(i))
            print("sentence: %s"%(sentence))
            print("input_ids:%s"%(input_ids))
            print("attention_masks:%s"%(input_mask))
            print("segment_ids:%s"%(segment_ids))
            print("\n")

    return full_input_ids , full_input_masks, full_segment_ids, trimmed_sentences_idx