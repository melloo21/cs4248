# Parameters
FILEPATH = "scicite_data/*.jsonl" # dataset path to read all jsonl files
MODEL_NAME = "xlnet-base-cased"
save_flag = False # To decide to save or not
output_dir = None # Need to fill this to save file
pretrained_dir = None # Fill up pretrained items

# Hyperparameters
max_token_length = 256
special_tokens= True
batch_num = 32
learning_rate = 3e-5
epochs = 2
eps =1e-8
num_warmup_steps = 0