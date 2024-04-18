import datetime, json, os, pickle, random, re, time, torch, transformers
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from tqdm import tqdm

from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler, Sampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, get_linear_schedule_with_warmup, utils
from bertviz import head_view, model_view

import sklearn_crfsuite
import scipy
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, ParameterGrid, RandomizedSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# get data
# data = json.load(open("data/processed/processed_text_balanced.json", "r"))
data = json.load(open("processed_text_balanced.json", "r"))
data = pd.DataFrame(data)
data_diag = json.load(open("processed_dialogues.json", "r"))
data_no_bal = json.load(open("processed_text.json", "r"))

# tokenizer
# model_path = "/home/users/nus/e1329380/scratch/models/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b" ###
model_path = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)

## for reference if want to freeze ##
# # freeze all of bert parameters, we only finetune the classifier layer (the FC layer)
# for param in model.bert.parameters():
#     param.requires_grad = False

# freeze the embeddings layer and the first 5 transformer layers
# modules = [model.bert.embeddings, *model.bert.encoder.layer[:6]] #Replace 5 by what you want
# for module in modules:
#     for param in module.parameters():
#         param.requires_grad = False

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class myDataset(Dataset):

    def __init__(self, X, y, tokenizer, max_len, label_mapper):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_mapper = label_mapper

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = str(self.X[idx])
        encoding = self.tokenizer.encode_plus(
            x,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = 'max_length',
            truncation = 'longest_first',
            return_attention_mask = True,
            return_token_type_ids = False,
            return_tensors = 'pt'
        )
        return{
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_mapper[self.y[idx]], dtype = torch.long)
        }

def create_data_loader(some_df, tokenizer, max_len, batch_size, label_mapper, purpose):
    ds = myDataset(
        X = some_df.raw.to_numpy(),
        y = some_df.y.to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len,
        label_mapper = label_mapper
    )
    if purpose == 'train':
        return DataLoader(
            ds,
            batch_size = batch_size,
            sampler = RandomSampler(ds),
            num_workers = 2
        )
    if purpose == 'val':
        return DataLoader(
            ds,
            batch_size = batch_size,
            sampler = SequentialSampler(ds),
            num_workers = 2
        )
    
def trainer(train_dataloader, validation_dataloader, epochs, model, criterion, optimizer, scheduler):
    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []
    best_f1 = 0
    best_model = None
    best_epoch = None

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Unpack this training batch from our dataloader.
            b_input_ids = batch['input_ids'].to("cuda")
            b_input_mask = batch['attention_mask'].to("cuda")
            b_labels = batch['labels'].to("cuda")

            # Perform a forward pass (evaluate the model on this training batch).
            y_pred = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            # Get the loss and "logits" output by the model.
            logits = y_pred.logits
            loss = criterion(logits, b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end.
            total_train_loss += loss.item()

            # Always clear any previously calculated gradients before performing a
            # backward pass.
            optimizer.zero_grad()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        
        if validation_dataloader:
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            total_eval_loss = 0
            predictions, true_labels = [], []

            # Evaluate data for one epoch
            for batch in validation_dataloader:

                # Unpack this training batch from our dataloader.
                b_input_ids = batch['input_ids'].to("cuda")
                b_input_mask = batch['attention_mask'].to("cuda")
                b_labels = batch['labels'].to("cuda")

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():

                    # Forward pass, calculate logit predictions.
                    y_val = model(input_ids=b_input_ids, attention_mask=b_input_mask)

                # Get the loss and "logits" output by the model.
                logits = y_val.logits
                loss = criterion(logits, b_labels)

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu()
                label_ids = b_labels.to('cpu')

                # Store predictions and true labels (only used to evaluate last epoch)
                predictions.extend(logits)
                true_labels.extend(label_ids)

            # Report the final accuracy for this validation run.
            predictions = torch.argmax(torch.stack(predictions), 1)
            true_labels = torch.stack(true_labels)
            f1 = f1_score(true_labels, predictions)
            print("  Valition F1 score: {0:.2f}".format(f1))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. F1 score.': f1,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_epoch = epoch_i + 1
            
        else:
            best_f1 = None
            best_epoch = None
            best_model = model
            training_stats.append(
                            {
                                'epoch': epoch_i + 1,
                                'Training Loss': avg_train_loss,
                                'Training Time': training_time
                            }
                        )
    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    print(f"Best Epoch: {best_epoch}, Best f1 val score: {best_f1}.")

    return training_stats, best_f1, best_epoch, best_model
    

def tuning(target, params):
    """
    hyperparameter tuning
    """
    epochs = 5 if "epochs" not in params else params["epochs"]
    label_mapper = {"neutral": 0, target: 1}
    
    hyperparam_combis = []
    validation_scores = []
    ## hyperparameter tuning
    for raw in params['raw']:
        for max_len in params['max_len']:
            for batch_size in params['batch_size']:
                for lr in params['lr']:
                    print("-"*100)
                    print(f"Doing {raw}_{max_len}_{batch_size}_{lr}...")
                    # train, dev
                    train = data[(data.split == "train") & (data.y.isin(["neutral", target]))].copy()
                    dev = data[(data.split == "dev") & (data.y.isin(["neutral", target]))].copy()
                    if not params["raw"]:
                        train["raw"] = train["clean"]
                        dev["raw"] = dev["clean"]
                    hyperparam_combis.append({'raw': raw, 'max_len': max_len, 'batch_size': batch_size, 'lr': lr})
                    # model
                    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
                    # dataloader
                    train_dataloader = create_data_loader(train, tokenizer, max_len, batch_size, label_mapper, 'train')
                    validation_dataloader = create_data_loader(dev, tokenizer, max_len, batch_size, label_mapper, 'val')
                    # loss
                    class_weights = compute_class_weight(class_weight="balanced", classes=np.array(range(2)), y=(train['y'].map(label_mapper)).tolist())
                    criterion = nn.CrossEntropyLoss(weight = torch.tensor(class_weights, dtype = torch.float)).to("cuda")
                    # optimizer and scheduler
                    optimizer = AdamW(model.parameters(),
                                      lr = lr,
                                      eps = 1e-8)

                    # Total number of training steps is [number of batches] x [number of epochs].
                    total_steps = len(train_dataloader) * epochs
                    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                                num_warmup_steps = total_steps * 0.05,
                                                                num_training_steps = total_steps)
                    _, best_f1, best_epoch, _ = trainer(train_dataloader = train_dataloader, validation_dataloader = validation_dataloader, epochs = epochs, model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler)
                    validation_scores.append(best_f1)
                    hyperparam_combis[-1]['epochs'] = best_epoch
    
    return hyperparam_combis, validation_scores

def evaluate(target, best_params, best_model):
    # test
    test = data[(data.split == "test") & (data.y.isin(["neutral", target]))].copy()
    if not best_params["raw"]:
        test["raw"] = test["clean"]
    label_mapper = {"neutral": 0, target: 1}
    test_dataloader = create_data_loader(test, tokenizer, best_params['max_len'], 128, label_mapper, 'val')
    
    # model
    best_model.eval()

    # Report the number of sentences.
    print('Number of test sentences: {:,}\n'.format(test.shape[0]))

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in test_dataloader:
        # Add batch to GPU
        # Unpack the inputs from our dataloader
        b_input_ids = batch['input_ids'].to("cuda")
        b_input_mask = batch['attention_mask'].to("cuda")
        b_labels = batch['labels'].to("cuda")

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            y_val = best_model(input_ids=b_input_ids, attention_mask=b_input_mask)

        logits = y_val.logits

        # Move logits and labels to CPU
        logits = logits.detach().cpu()
        label_ids = b_labels.to('cpu')

        # Store predictions and true labels
        predictions.extend(logits)
        true_labels.extend(label_ids)

    predictions = torch.argmax(torch.stack(predictions), 1)
    true_labels = torch.stack(true_labels)

    print('DONE.')

    ### Can edit your evaluation metric here
    f1 = f1_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names = ["neutral", target])    
    return f1, report

def main():
    """
    nohup python trainer.py >> logs/anger_log.txt &
    nohup python trainer.py >> logs/joy_log.txt &
    nohup python trainer.py >> logs/surprise_log.txt &
    nohup python trainer.py >> logs/sadness_log.txt &
    nohup python trainer.py >> logs/fear_log.txt &
    nohup python trainer.py >> logs/disgust_log.txt &
    """
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"]="3"
    
    RANDOM_SEED = 2024
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    target = "disgust" ### Change accordingly: "anger", "joy", "surprise", "sadness", "fear", "disgust"
    params = {
        "raw": [True, False],
        "max_len": range(60,71,10),
        "batch_size": [32, 64, 128],
        "lr": [3e-4, 1e-4, 5e-5],
    }
    hyperparam_combis, validation_scores = tuning(target, params)
    best_params = hyperparam_combis[np.argmax(validation_scores)]
    print(f"\nBest Parameters are:\n{best_params}\n")

    epochs = best_params["epochs"]
    # train, dev
    train = data[((data.split == "train")|(data.split == "dev")) & (data.y.isin(["neutral", target]))].copy()
    if not best_params["raw"]:
        train["raw"] = train["clean"]
    label_mapper = {"neutral": 0, target: 1}
    # model
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
    # dataloader
    train_dataloader = create_data_loader(train, tokenizer, best_params["max_len"], best_params["batch_size"], label_mapper, 'train')
    # loss
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array(range(2)), y=(train['y'].map(label_mapper)).tolist())
    criterion = nn.CrossEntropyLoss(weight = torch.tensor(class_weights, dtype = torch.float)).to("cuda")
    # optimizer and scheduler
    optimizer = AdamW(model.parameters(),
                      lr = best_params["lr"],
                      eps = 1e-8)

    # Total number of training steps is [number of batches] x [number of epochs].
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = total_steps * 0.05,
                                                num_training_steps = total_steps)

    _, _, _, best_model = trainer(train_dataloader = train_dataloader, validation_dataloader = None, epochs = epochs, model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler)
    f1, report = evaluate(target, best_params, best_model)
    print(f"\nTest F1 score is {f1}\n")
    print(f"Test Classification Report is:\n{report}")
    
if __name__ == "__main__":
    main()