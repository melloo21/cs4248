import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class CreateDataloader:
  def __init__(
      self,
      input,
      seg,
      mask,
      tags,
      loader_type
  ):
  # Convert data to tensor
    self.seg = seg
    print(self.seg, seg)
    self.mask = mask
    self.input = input
    self.tags = tags

    self.loader_type = loader_type
    self.total_size = len(self.input)


  def _data_loader(self, batch_num:int):
    input_tensor = torch.tensor(self.input)
    input_mask = torch.tensor(self.mask)
    input_seg = torch.tensor(self.seg)
    input_tags = torch.tensor(self.tags)

    # Creating Tensor Dataset
    self.tensor_dataset = TensorDataset(input_tensor, input_mask, input_seg, input_tags)

    if self.loader_type.lower() == "train":
      sampler = RandomSampler(self.tensor_dataset)
      return DataLoader(self.tensor_dataset, sampler=sampler, batch_size=batch_num,drop_last=True)
    else:
      sampler = SequentialSampler(self.tensor_dataset)
      return DataLoader(self.tensor_dataset, sampler=sampler, batch_size=batch_num)
