import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class CreateDataset:
  def __init__(
      self,
      filepath,
      columns_required=["label", "string"],
      tag2idx={'0': 0,'1': 1, '2':2}
    ):
    # Tag to index
    self.tag2idx = tag2idx
    self.filepath = filepath
    self.columns_required = columns_required
    self._get_all_json_files()
    self._create_datasets()
    self.classes = self.df_train.label.unique()

  def _get_all_json_files(self):

    all_scicite = dict()
    for g in glob.glob(self.filepath):
      name = g.split("/")[-1].split(".")[0]
      all_scicite[name] = pd.read_json(path_or_buf=g, lines=True)

      # Filter based on required columns
      all_scicite[name] = all_scicite[name][self.columns_required]

      # Map labels to correct class labels (numerical)
      le = LabelEncoder()
      le.fit(all_scicite[name].label)
      all_scicite[name].loc[:,"encoded_label"] = le.transform(all_scicite[name].label)

    self.all_scicite = all_scicite

  def _create_datasets(self):
    self.df_train = self.all_scicite["train"]
    self.df_test = self.all_scicite["test"]
    self.df_dev = self.all_scicite["dev"]

  def _get_sentence_data(self, dataset):
    return self.all_scicite[dataset].string.to_list()

  def _label_to_id(self, dataset):
    return [self.tag2idx[str(lab)] for lab in self.all_scicite[dataset].encoded_label]

  def _index_tag(self, dataset):
    return {self.tag2idx[key] : key for key in self.tag2idx.keys()}
