import numpy as np 
import pandas as pd 
from preprocessing import *

from sklearn.naive_bayes import GaussianNB, MultinomialNB # MutinominalNB
from sklearn.svm import SVC 
# from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
# from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW

def models(X, y, option = 0, val_size=0.2, random_state=0):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)

    time_init = time.time()
    if option == 0:
        model = KNeighborsClassifier()
        model = model.fit(X_train, y_train)

    elif option == 1:
        model = GaussianNB()
        model = model.fit(X_train, y_train)
    
    elif option == 2:
        model = SVC()
        model = model.fit(X_train, y_train)

    elif option == 3:
        model = DecisionTreeClassifier()
        model = model.fit(X_train, y_train)

    else:
        model = LogisticRegression()
        model = model.fit(X_train, y_train)

    start = time.time()
    y_pred_train = model.predict(X_train)
    stop_train = time.time()
    y_pred_val = model.predict(X_val)
    stop_val = time.time()

    # Report
    print(f'Report of train set with {int(len(X_train))} examples: ')
    print(f'Time to train: {round(start-time_init, 4)}s')
    print(f'Time to predict train set: {round(stop_train-start, 4)}s')
    print(classification_report(y_train, y_pred_train))
    print('\nReport of validation set:')
    print(f'Time to predict validation set {round(stop_val-start, 4)}s')
    print(classification_report(y_val, y_pred_val))

    return model

def cat_models(X, y, option = 0, val_size=0.2, random_state=0):
    pipeline = Pipeline([('count',CountVectorizer()),('tfidf',TfidfTransformer())])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
    X_train = pipeline.fit_transform(X_train).toarray()
    X_val = pipeline.transform(X_val).toarray()

    time_init = time.time()

    if option == 0:
        model = KNeighborsClassifier()
        model = model.fit(X_train, y_train)

    elif option == 1:
        model = GaussianNB()
        model = model.fit(X_train, y_train)
    
    elif option == 2:
        model = SVC()
        model = model.fit(X_train, y_train)

    elif option == 3:
        model = DecisionTreeClassifier()
        model = model.fit(X_train, y_train)

    else:
        model = LogisticRegression()
        model = model.fit(X_train, y_train)

    start = time.time()
    y_pred_train = model.predict(X_train)
    stop_train = time.time()
    y_pred_val = model.predict(X_val)
    stop_val = time.time()

    # Report
    print(f'Report of train set with {int(len(X_train))} examples: ')
    print(f'Time to train: {round(start-time_init, 4)}s')
    print(f'Time to predict train set: {round(stop_train-start, 4)}s')
    print(classification_report(y_train, y_pred_train))
    print('\nReport of validation set:')
    print(f'Time to predict validation set {round(stop_val-start, 4)}s')
    print(classification_report(y_val, y_pred_val))

    return model, pipeline

def predict(model, X, y, name_set='test set'):
    start = time.time()
    y_pred = model.predict(X)
    stop = time.time()

    print(f'Time to predict on the {name_set} with {len(X)} examples: {round(stop-start, 4)}s')
    print(classification_report(y, y_pred))
    
    return y_pred

def cat_predict(model, pipeline, X, y, name_set='test set'):
    X = pipeline.transform(X).toarray()
    
    start = time.time()
    y_pred = model.predict(X)
    stop = time.time()

    print(f'Time to predict on the {name_set} with {len(X)} examples: {round(stop-start, 4)}s')
    print(classification_report(y, y_pred))
    
    return y_pred


##################################################################################################
############################################### BERT ############################################
# Create model
# from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW

def create_model(path_config="PhoBERT_base_transformers/config.json", path_model="PhoBERT_base_transformers/model.bin"):
  # from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW
  config = RobertaConfig.from_pretrained(
    path_config, from_tf=False, num_labels = 2, output_hidden_states=False,
  )
  BERT = RobertaForSequenceClassification.from_pretrained(
    path_model,
    config=config
  )
  BERT.cuda()
  print('Done')
  return BERT

# Evaluate
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    F1_score = f1_score(pred_flat, labels_flat, average='macro')
    cm = confusion_matrix(pred_flat, labels_flat)
    
    return accuracy_score(pred_flat, labels_flat), F1_score, cm

# Train model
import random
from tqdm import tqdm_notebook

def bert_train(train_sents, train_labels, test_text, test_labels, device = 'cuda', epochs = 10, path="./models/Bert_classification_balance_data_v1.pt", w2v=2):
  train_dataloader, val_dataloader, test_dataloader, train_sents, val_sents, train_labels, val_labels, vectorizer = bert_preprocessing(train_sents, train_labels, test_text, test_labels, w2v=w2v)
  BERT = create_model()
  param_optimizer = list(BERT.named_parameters())
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]

  optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)


  for epoch_i in range(0, epochs):
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      print('Training...')

      total_loss = 0
      BERT.train()
      train_accuracy = 0
      nb_train_steps = 0
      train_f1 = 0
      
      for step, batch in tqdm_notebook(enumerate(train_dataloader)):
          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_labels = batch[2].to(device)

          BERT.zero_grad()
          outputs = BERT(b_input_ids, 
              token_type_ids=None, 
              attention_mask=b_input_mask, 
              labels=b_labels)
          loss = outputs[0]
          total_loss += loss.item()
          
          logits = outputs[1].detach().cpu().numpy()
          label_ids = b_labels.to('cpu').numpy()
          tmp_train_accuracy, tmp_train_f1, cm = flat_accuracy(logits, label_ids)
          train_accuracy += tmp_train_accuracy
          train_f1 += tmp_train_f1
          nb_train_steps += 1
          
          loss.backward()
          torch.nn.utils.clip_grad_norm_(BERT.parameters(), 1.0)
          optimizer.step()
          
      avg_train_loss = total_loss / len(train_dataloader)
      print(" Accuracy: {0:.4f}".format(train_accuracy/nb_train_steps))
      print(" F1 score: {0:.4f}".format(train_f1/nb_train_steps))
      # print(cm)
      print(" Average training loss: {0:.4f}".format(avg_train_loss))

      print("Running Validation...")
      BERT.eval()
      eval_loss, eval_accuracy = 0, 0
      nb_eval_steps, nb_eval_examples = 0, 0
      eval_f1 = 0
      for batch in tqdm_notebook(val_dataloader):

          batch = tuple(t.to(device) for t in batch)

          b_input_ids, b_input_mask, b_labels = batch

          with torch.no_grad():
              outputs = BERT(b_input_ids, 
              token_type_ids=None, 
              attention_mask=b_input_mask)
              logits = outputs[0]
              logits = logits.detach().cpu().numpy()
              label_ids = b_labels.to('cpu').numpy()

              tmp_eval_accuracy, tmp_eval_f1, cm = flat_accuracy(logits, label_ids)

              eval_accuracy += tmp_eval_accuracy
              eval_f1 += tmp_eval_f1
              nb_eval_steps += 1
      print(" Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))
      print(" F1 score: {0:.4f}".format(eval_f1/nb_eval_steps))
      # print(cm)
  print("Training complete!\n")
  # Report
  train_labels_pred = bert_predict(train_sents, BERT, vectorizer, w2v=w2v)
  val_labels_pred = bert_predict(val_sents, BERT, vectorizer, w2v=w2v)
  report(train_labels_pred, train_labels, name='Train')
  print('*'*100)
  report(val_labels_pred, val_labels, name='validation')
  print('Save model: ', path)
  torch.save(BERT, path)

  return BERT, vectorizer

def bert_predict(sents, model, vectorizer, w2v='bpe_tfidf'):
  tfidf = vectorizer.transform(sents).toarray()
  if w2v=='bpe_tfidf':
    ids = convert_line(sents, MAX_LEN=128)
    ids = np.concatenate((ids, tfidf*100), axis=1)
  elif w2v=='bpe':
    ids = convert_line(sents, MAX_LEN=256)
  elif w2v == 'tfidf':
    ids = tfidf*100
  else:
    print('Enter again')
    return _
    
  mask = create_masks(ids)
  loader = Load_dataloader(ids,np.zeros(ids.shape[0]),mask)
  pred_labels = []
  ## Predict
  device = 'cuda'
  for batch in loader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
      outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]

    logits = logits.detach().cpu().numpy()
    arr = np.argmax(logits, axis=1).flatten()
    # arr = np.max(logits, axis=1)
    for i in arr:
      pred_labels.append(i)
      
  return pred_labels

def report(y_pred, y_label, name='Test'):
  
  print(f'Report for {name} set:')
  print(classification_report(y_label, y_pred))
  print('\nConfusion maxtrix:')
  print(confusion_matrix(y_label, y_pred))