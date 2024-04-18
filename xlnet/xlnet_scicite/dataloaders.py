from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class CreateDataloaders:
  def __init__(self, batch_num):
    self.batch_num =batch_num

  def get_train_loader(self,input_ids, attention_masks, labels):
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    dataloader = DataLoader(
                dataset,  # The training samples.
                sampler = RandomSampler(dataset), # Select batches randomly
                batch_size = self.batch_num # Trains with this batch size.
            )
    return dataloader

  def get_eval_dataloader(self,input_ids, attention_masks, labels):
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(
                    dataset, 
                    sampler=SequentialSampler(dataset), 
                    batch_size=self.batch_num
                    )

    return dataloader