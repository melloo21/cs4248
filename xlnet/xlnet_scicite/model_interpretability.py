import lime
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Change paths
dir = None # Change to get pretrained
save_path = None

xlnet_loaded = AutoModelForSequenceClassification.from_pretrained(dir)
xlnet_tokenizer = AutoTokenizer.from_pretrained(dir)
class_names = ['background','method', 'results']

def predictor(texts):
  outputs = xlnet_loaded(**xlnet_tokenizer(texts, return_tensors="pt", padding=True).to(device))
  probas = F.softmax(outputs.logits).cpu().detach().numpy()
  return probas

explainer = LimeTextExplainer(class_names=class_names)

for key ,value in plot_cases.items():
  print(f"index {key}")
  str_to_predict = test_sentence[key]
  exp = explainer.explain_instance(str_to_predict, predictor, num_features=10, num_samples=100,  labels=[value])
  
  # Show in notebook
  exp.show_in_notebook(text=str_to_predict)

  # Save html refer to plots
  exp.save_to_file(f'{save_path}test_{key}.html', predict_proba=True, show_predicted_value=True,labels=[value])
  
  fig = exp.as_pyplot_figure(label=value)
  fig.savefig(f'{save_path}/test_{key}.jpg')