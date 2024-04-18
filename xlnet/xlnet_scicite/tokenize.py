# Tokenizer
import torch
from transformers import AutoTokenizer

# Load Tokenizer 
class CreateTokens:
  def __init__(self, max_length:int, special_tokens:bool):
    self.max_length = max_length
    self.special_tokens = special_tokens
  
  def xlnettokenize(self, all_sentences,labels, random_print:int=0):
    tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')

    # Return input ids and attention masks
    input_ids = []
    attention_masks = []
    # For every sentence
    for sentence in all_sentences:
      encoded_dict = tokenizer.encode_plus(
                          sentence,
                          add_special_tokens = self.special_tokens,
                          truncation = True,
                          max_length = self.max_length,
                          pad_to_max_length = True,
                          return_attention_mask = True,
                          return_tensors = 'pt',
                    )

      input_ids.append(encoded_dict['input_ids'])
      attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    # Printing a random sample
    print("len :", len(input_ids[random_print]), len( all_sentences[random_print]))
    print('Original: ', all_sentences[random_print])
    print('Token IDs:', input_ids[random_print])

    return input_ids, attention_masks, labels