# %%
"""
The code is divided into three blocks:
    Data Preprocessing
        Functions:
            handleTextCleaning(): For text cleaning purpose
            handleDataPreprocessing(): For processing dataset before feeding to the model
    
    Model Building
        Class:
            SentencePairDataset(): Prepares dataset for the dataloader
        Functions:
            handleModelTraining(): Trains the model
            handleModelEvaluation(): Validates the training
            handleModelTesting(): Tests the model
            handleTrainEvalModel(): Contains Train validation loop
            buildModel(): Encapsulate the entire process
    Result Analysis
"""
# I have tested using three different model:
# bert: 'bert-base-uncased'
# biobert: 'emilyalsentzer/Bio_ClinicalBERT'
# electra: 'google/electra-base-discriminator'
# each functions and it's working has been described in it's respective cell
               
            

# %%
# Do not run this cell if you have installed the environment or have already installed the packages
# !pip install transformers torch

# %%
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm.notebook import tqdm
import utils
import torch
import logging
import re
import sys
import nltk
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

# removing all existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# creating handlers
stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(f'../logs/roberta_logfile.log')

# creating formatters
formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# adding handlers to the logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.propagate = False

logger.info("Starting...")

# %%
class SentencePairDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data = self.dataframe.iloc[idx]
        id=data['id']
        sentence1 = data['sentence1']
        sentence2 = data['sentence2']
        label = data['label']
        inputs = self.tokenizer.encode_plus(
            sentence1, sentence2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'id': id,
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# %% [markdown]
# Data Preprocessing

# %%
def handleTextCleaning(text):
    """
    A function to clean text data, preserving biomedical terms.

    Parameters:
    - text (str): The text to be cleaned.

    Returns:
    - str: The cleaned text.
    """
    # Converting text to lowercase (optional, may not be ideal for case-sensitive terms)
    text = text.lower()

    # Removing newline characters
    text = text.replace('\n', ' ')
    text = text.replace('_', ' ')

    # Removing extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# %%
def handleDataPreprocessing(data_path):
    """
    Preprocesses data before feeding to the model: focusing on text cleaning and dataset splitting.

    The function performs several key operations:
    - Reads data from the specified CSV file.
    - Extracts and retains important columns relevant to the task: 'PREDICATION_ID', 'PREDICATE', 'SUBJECT_TEXT', 'OBJECT_TEXT', 'SENTENCE', and 'LABEL'.
    - Adds new columns to the DataFrame for holding cleaned text data: 'PREDICATION', 'PREDICATION_clean', and 'SENTENCE_clean'.
    - Constructs the 'PREDICATION' column by concatenating 'SUBJECT_TEXT', 'PREDICATE', and 'OBJECT_TEXT'.
    - Cleans text data in 'PREDICATION' and 'SENTENCE' columns using the 'handleTextCleaning' function, and stores the results in 'PREDICATION_clean' and 'SENTENCE_clean'.
    - Converts the 'LABEL' column into a binary format, mapping 'y' to 1 and 'n' to 0.
    - Selects and renames columns for consistency and clarity, resulting in a DataFrame with 'id', 'sentence1', 'sentence2', and 'label' columns.
    - Splits the sampled data into training, development, and test sets.

    Parameters:
    - data_path (str): A string representing the path to the raw CSV data file that needs to be preprocessed.

    Returns:
    - tuple:    A tuple containing three DataFrames (train_df, dev_df, test_df), representing the training, development, and test sets, respectively.
                Each DataFrame includes cleaned and processed text data along with the respective 'split' label indicating its set designation.
    """

    data=pd.read_csv(data_path)
    print(data.head())
    # extracting relevant columns
    data=data[["PREDICATION_ID","PREDICATE","SUBJECT_TEXT","OBJECT_TEXT","SENTENCE","LABEL"]]
    print(data.head())
    # inserting new 'PREDICATION'(for storing concatenated string <'SUBJECT_TEXT' + 'PREDICATE' + 'OBJECT_TEXT'>),
    # 'PREDICATION_clean' and 'SENTENCE_clean' columns for storing cleaned predication and sentences
    data.insert(loc=4, column='PREDICATION', value=0)
    data.insert(loc=5, column='PREDICATION_clean', value=0)
    data.insert(loc=7, column='SENTENCE_clean', value=0)
    for i in tqdm(range(0,len(data))):
        #concatenating 'SUBJECT_TEXT', 'PREDICATE', 'OBJECT_TEXT' and saving into "PREDICATION" column
        data.iloc[i,4]=data.iloc[i,2]+" "+data.iloc[i,1]+" "+data.iloc[i,3]
        #cleaning both "PREDICATION" and "SENTENCE" column
        data.iloc[i,5]=handleTextCleaning(data.iloc[i,4])
        data.iloc[i,7]=handleTextCleaning(data.iloc[i,6])
    # converting labels into binary format
    data['LABEL'] = data['LABEL'].map({'y': 1, 'n': 0})
    all_dataset=data[["PREDICATION_ID","SENTENCE_clean","PREDICATION_clean", "LABEL"]]
    # renaming and taking columns ('id', 'sentence1', 'sentence2', and 'label')
    all_dataset=all_dataset.rename(columns={"PREDICATION_ID":"id", "SENTENCE_clean":"sentence1", "PREDICATION_clean":"sentence2", "LABEL":"label"})
    all_dataset=all_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    # splitting full dataset into train, valid and test set having 0.6:0.2:0.2 ratio
    train_df, temp_df = train_test_split(all_dataset, test_size=0.4, random_state=42, stratify=all_dataset['label'])
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    train_df['split'] = 'train'
    dev_df['split'] = 'dev'
    test_df['split'] = 'test'

    # Concatenate the datasets back together
    df = pd.concat([train_df, dev_df, test_df])
    print(df.head())
    logger.info(f"train,eval,test splits: {train_df.shape[0]},{ dev_df.shape[0]},{test_df.shape[0]}")
    return train_df, dev_df, test_df

# %% [markdown]
# Model Building

# %%
# Training Function
def handleModelTraining(model, data_loader, optimizer, scheduler, device, n_samples):
    """
    Trains a machine learning model using the provided data, optimizer, and scheduler.

    This function goes through the following steps:
    - Sets the model to training mode.
    - Iterates over batches of data in 'data_loader':
        -- Moves input data and labels to the specified 'device' (e.g., GPU or CPU).
        -- Passes the input data to the model and computes the output.
        -- Calculates the loss between the model output and the true labels.
        -- Performs backpropagation to compute gradients.
        -- Updates model parameters using the optimizer.
        -- Updates the learning rate using the scheduler.
        -- Resets gradients in the optimizer for the next iteration.

    Parameters:
    - model (torch.nn.Module): The model to be trained.
    - data_loader (DataLoader): An iterable DataLoader containing the training data.
    - optimizer: The optimization algorithm used to update the model's weights.
    - scheduler: Learning rate scheduler to adjust the learning rate during training.
    - device (torch.device): The device (e.g., 'cuda', 'cpu') on which to perform computations.
    - n_samples (int): Total number of samples in the dataset.

    Returns:
    - tuple: A tuple containing the training accuracy and the average loss per example.
    """
    model = model.train()
    losses = []
    correct_predictions = 0
    # iterating over batches of data in 'data_loader'
    for batch in tqdm(data_loader):
        # moving input_ids, attention_mask, labels to available device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # getting output from the model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss.mean()
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        # performing backpropagation
        loss.backward()
        #  clipping gradients to prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #updating optimizer and scheduler
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_samples, np.mean(losses)


# %%
# Validation Function
def handleModelEvaluation(model, data_loader, device, n_samples):
    """
    Evaluates the model on the validation dataset.

    This function performs model evaluation by calculating loss, accuracy, AUC (Area Under Curve),
    and generating a confusion matrix, sensitivity, specificity, and classification report.
    It operates in a no-gradient context to optimize memory and computation during evaluation.

    Steps performed:
    - Sets the model to evaluation mode.
    - Iterates over the data_loader to process data in batches.
        -- Moves the input data and labels to the specified device (e.g., GPU).
        -- Computes the model's outputs based on the input data.
        -- Calculates and accumulates the loss.
        -- Extracts logits from the model outputs and calculates the probabilities using softmax.
    - Determines the predicted labels and counts the number of correct predictions.
    - Stores all results (loss, predictions, probabilities, true labels)
    - Computes the AUC score using true labels and predicted probabilities.
    - Generates and logs the confusion matrix, sensitivity (True Positive Rate) and specificity (True Negative Rate).
    - Generates and logs a classification report containing precision, recall, and F1-score.
    - Returns the accuracy, average loss, AUC, sensitivity, and specificity.

    Parameters:
    - model (torch.nn.Module): The model to be evaluated.
    - data_loader (DataLoader): DataLoader object that provides batches of data for evaluation.
    - device (torch.device): The computing device (e.g., 'cuda', 'cpu') on which the model runs.
    - n_samples (int): The total number of samples in the validation dataset.

    Returns:
    - tuple: A tuple containing the accuracy, average loss, AUC, sensitivity, and specificity of the model on the evaluation dataset.
    """

    # Setting the model to evaluation mode.
    model = model.eval()
    losses = []
    correct_predictions = 0
    true_labels = []
    predictions_labels = []
    probabilities = []
    # disabling gradient calculation
    with torch.no_grad():
        # iterating over batches of data in 'data_loader'
        for batch in tqdm(data_loader):
            # moving input_ids, attention_mask, labels to available device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # getting output from the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            # extracting raw probabilities for calculating AUC
            prob = F.softmax(logits, dim=1)
            # computing predicted label
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.mean().item())
            predictions_labels.extend(preds.cpu().numpy())
            probabilities.extend(prob.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    true_labels = np.array(true_labels)
    predictions_labels = np.array(predictions_labels)
    # taking the probability of the positive class
    proba = np.array(probabilities)[:, 1]
    # calculating and logging AUC
    auc = roc_auc_score(true_labels, proba)
    logger.info("----- Validation Results -----")
    logger.info(f"AUC: {auc}")
    # calculating and logging Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predictions_labels)
    logger.info(f"Confusion Matrix:\n {conf_matrix}")
    TN, FP, FN, TP = conf_matrix.ravel()

    # calculating Sensitivity, Specificity and Classification Report
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    recall = sensitivity
    f1_score = 2 * (precision * recall) / (precision + recall)
    class_report = classification_report(true_labels, predictions_labels)
    logger.info(class_report)
    return correct_predictions.double() / n_samples, np.mean(losses), auc, f1_score, sensitivity, specificity, precision

# %%
def handleModelTesting(set, model_name, model, test_df, device):
    """
    Test the model on the test dataset and logs performance metrics.

    This function performs the following steps:
    - Sets the model to evaluation mode.
    - Creates a dataset and dataloader for the test data.
    - Iterates over the test dataset in batches, performing predictions with the model.
        -- Stores batch ids, predicted labels, probabilities, and true labels.
    - Calculates various performance metrics, including AUC, accuracy, sensitivity, specificity, and confusion matrix.
    - Creates a DataFrame with test results, including true labels, predictions, and probabilities for both classes.
    - Saves the test results DataFrame to a CSV file for further analysis.

    Parameters:
    - set (int): An identifier for the test set, used in the filename of the saved results.
    - model_name (str): The name of the model used for loading the tokenizer.
    - model (torch.nn.Module): The PyTorch model to be evaluated.
    - test_df (pd.DataFrame): A DataFrame containing the test data.
    - device (torch.device): The PyTorch device (e.g., 'cuda' or 'cpu') to which the model and data are moved.

    Returns:
    - tuple: A tuple containing the calculated metrics: accuracy, AUC, sensitivity, and specificity.
    """

    # setting the model to evaluation mode
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")

    
    test_dataset = SentencePairDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    ids=[]
    true_labels = []
    predictions_labels = []
    probabilities = []
    # disabling gradient calculation
    with torch.no_grad():
        # iterating over batches of data in 'data_loader'
        for batch in test_loader:
            # moving input_ids, attention_mask, labels to available device
            id=batch['id']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # getting output from the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            logits = outputs.logits
            # extracting raw probabilities for calculating AUC
            prob = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, dim=1)
            # also preserving id to track the output result for future analysis
            ids.extend(id)
            predictions_labels.extend(preds.cpu().numpy())
            probabilities.extend(prob.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    true_labels = np.array(true_labels)
    proba = np.array(probabilities)[:, 1]
    predictions_labels = np.array(predictions_labels)
    # calculating and logging AUC
    auc = roc_auc_score(true_labels, proba)
    logger.info("----- Test Results -----")
    logger.info(f"AUC: {auc}")
    # calculate Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predictions_labels)

    TN, FP, FN, TP = conf_matrix.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Confusion Matrix:\n {conf_matrix}")
    # calculating Sensitivity and Specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision= TP / (TP + FP)
    recall = sensitivity
    f1_score = 2 * (precision * recall) / (precision + recall)
    # calculate Classification Report
    class_report = classification_report(true_labels, predictions_labels)
    logger.info(class_report)
    # Saving result of each run for future analysis
    test_result_df=pd.DataFrame({"id":ids,
                                 "y_true":true_labels,
                                 "y_pred": predictions_labels,
                                 "y_pred_0":np.array(probabilities)[:, 0],
                                 "y_pred_1":np.array(probabilities)[:, 1]
                                 })
    test_result_df.to_csv(f"../results/{model_name}_test_set_{set}_results.csv", index=False)
    return accuracy, auc, f1_score, sensitivity, specificity, precision

# %%
def handleTrainEvalModel(train_df, valid_df, model_name, num_epoch, batch_size, lr):
    """
    Trains and evaluates the model.

    This function performs several key steps:

    - Initializes the tokenizer and datasets for training and validation using the BERT model.
    - Creates DataLoader objects for the training and validation datasets with specified batch sizes.
    - Initializes the BERT model for our semrep task.
    - Sets up the model to use GPU if available, and configures multi-GPU training if multiple GPUs are detected.
    - Implements the optimizer (AdamW) a learning rate scheduler with a linear warm-up phase.
    - Trains the model for the specified number of epochs, logging training accuracy and loss.
    - Evaluates the model on the validation set after each epoch, logging accuracy, loss, AUC, sensitivity, and specificity.
    - Saves the model with the best validation loss observed during training.
    - Logs the best loss and the epoch at which it was found.

    Parameters:
    - train_df (DataFrame): The training data as a Pandas DataFrame.
    - valid_df (DataFrame): The validation data as a Pandas DataFrame.
    - model_name (str): The name of the pre-trained BERT model to be used.
    - num_epoch (int): The number of epochs for training the model.
    - batch_size (int): The batch size for training and validation.
    - lr (float): The learning rate for the optimizer.

    Returns:
    - tuple: A tuple containing the best validation accuracy, loss, AUC, epoch number, sensitivity, and specificity.
    """
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)


    num_gpus=torch.cuda.device_count()

        #defining our pretrained model
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion", ignore_mismatched_sizes=True,num_labels=2)

    # moving model to available device
    device=torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
    # defining tokenizer for processing model input
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")



    # SentencePairDataset is a customized class of Dataset object which not only handle tokenization of sentence pair
    # but also retains id column to track them for further analysis
    train_dataset = SentencePairDataset(train_df, tokenizer)
    val_dataset = SentencePairDataset(valid_df, tokenizer)

    # defining training and validation dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



    # parallel processing for speeding up the training process
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


    criterion = nn.CrossEntropyLoss() # can handle both binary and multiclass classification
    # defining the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * num_epoch
    # creating the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0.1*total_steps,
                                                num_training_steps=total_steps)
    epochs=num_epoch
    best_loss=1e5
    # training and evaluation loop
    for epoch in range(epochs):
        logger.info(f'Epoch {epoch + 1}/{epochs}')
        logger.info('-' * 10)

        # handleModelTraining function trains the model on entire trining set and returns trainng accuracy and loss
        train_acc, train_loss = handleModelTraining(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            len(train_dataset)
        )

        logger.info(f'Train loss {train_loss} accuracy {train_acc}')
        # handleModelEvaluation function evaluates the model on validation set and returns validation accuracy,loss and other metrices
        val_acc, val_loss, val_auc,val_f1, sensitivity, specificity, precision = handleModelEvaluation(
            model,
            val_loader,
            device,
            len(val_dataset)
        )
        model.train()

        # always looking for the least loss. saving the validation result if the loss is lower than the previous
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch+1
            best_acc= val_acc
            best_auc= val_auc
            best_f1= val_f1
            best_sensitivity = sensitivity
            best_specificity = specificity
            best_precision= precision
            logger.info(f"best_loss till now: {best_loss}")
            logger.info("saving the model....")
            # saving the model for testing it on test data
            torch.save(model,f"../models/semrep_simple_{model_name}_model")
        logger.info(f'Val loss {val_loss}, accuracy {val_acc}')
    logger.info(f"Best loss {best_loss} found at epoch {best_epoch}")
    return best_acc, best_auc,best_f1, best_loss, best_epoch, best_sensitivity, best_specificity, best_precision

# %%
def buildModel(model_name, lr, num_epoch, batch_size, num_runs, train_df, valid_df, test_df):
    """
    Trains, validates and tests the model across multiple runs,
    tracking and storing performance metrics on both validation and test datasets from each run.
    It executes the following steps:
    - Initializes dictionaries ('metrics_best' and 'metrics_test') to store performance metrics
       like Accuracy, AUC, Sensitivity, Specificity, Loss, and Epoch number.
    - Iterates over the specified number of runs ('num_runs'), logging the run number for tracking.
        --  In each run, it trains and validates the model using the 'handleTrainEvalModel' function
        --  Stores the best validation metrics from each run in 'metrics_best'.
        --  Loads the trained model and evaluates it on the test dataset using 'handleModelTesting'.
        --  Appends test metrics to 'metrics_test'.
    - After all runs, compiles the collected metrics into DataFrames ('val_result_df' and 'test_result_df')
    - Saves and returns these DataFrames to CSV files for later analysis.


    Parameters:
    - model_name (str): Name of the model to be used.
    - lr (float): Learning rate for the model training.
    - num_epoch (int): Number of epochs for each training run.
    - batch_size (int): Batch size for training the model.
    - num_runs (int): Number of runs for training and evaluation.
    - train_df (DataFrame): DataFrame containing the training data.
    - valid_df (DataFrame): DataFrame containing the validation data.
    - test_df (DataFrame): DataFrame containing the test data.

    Returns:
    - tuple: A tuple containing two DataFrames (val_result_df, test_result_df), representing the
             validation and test results across all runs.
    """
    # initializing dictionaries for for storing validation and test metrics
    metrics_best = {
        "Accuracy": [],
        "AUC": [],
        "F1":[],
        "Loss": [],
        "Epoch": [],
        "Sensitivity": [],
        "Specificity": [],
        "Precision": []
    }

    metrics_test = {
        "Accuracy": [],
        "AUC": [],
        "F1":[],
        "Sensitivity": [],
        "Specificity": [],
        "Precision": []
    }

    # iterating through multiple runs
    for i in tqdm(range(num_runs)):
        logger.info(f"########## RUN {i} ##########")

        # training and validation
        best_metrics = handleTrainEvalModel(train_df, valid_df, model_name, num_epoch, batch_size, lr)
        for metric_name, value in zip(metrics_best.keys(), best_metrics):
            metrics_best[metric_name].append(value.cpu().numpy() if metric_name == "Accuracy" else value)

        # loading model and running test
        loaded_model = torch.load(f"../models/semrep_simple_{model_name}_model")
        loaded_model.to('cuda' if torch.cuda.is_available() else 'cpu')

        test_metrics = handleModelTesting(i, model_name, loaded_model, test_df, loaded_model.module.device)
        for metric_name, value in zip(metrics_test.keys(), test_metrics):
            metrics_test[metric_name].append(value)

    # creating and saving result dataframes
    val_result_df = pd.DataFrame(metrics_best)
    test_result_df = pd.DataFrame(metrics_test)

    val_result_df.to_csv(f"../results/val_{model_name}_results.csv")
    test_result_df.to_csv(f"../results/test_{model_name}_results.csv")

    return val_result_df, test_result_df

# %%
MODEL_NAME="cardiffnlp/twitter-roberta-base-emotion"
KEY="roberta"
LR=2e-5
NUM_EPOCHS=12
BATCH_SIZE=16
NUM_RUNS=5
DATA_PATH="../data/substance_interactions.csv"

train_df, val_df, test_df=handleDataPreprocessing(DATA_PATH)
# train_df, val_df, test_df=handleDataPreparation(df)
val_result_df, test_result_df=buildModel(KEY,
                                         LR,
                                         NUM_EPOCHS,
                                         BATCH_SIZE,
                                         NUM_RUNS,
                                         train_df,
                                         val_df,
                                         test_df
                                         )


# %%
val_result_df.head()

# %%
test_result_df.head()

# %%
val_result_df.mean()

# %%
test_result_df.mean()

# %%
highest_AUC_row=test_result_df['AUC'].idxmax()
ana_df=pd.read_csv(f"../results/{KEY}_test_set_{highest_AUC_row}_results.csv")
ana_df.head()

# %%
utils.getAUCCurve(df=ana_df,name=f"{KEY}_roc_curve")

# %% [markdown]
# # **Analyzing the results (Verbal and All)**

# %%
src_df=pd.read_csv("../data/substance_interactions.csv")
src_df.head()

# %%
joined_df = ana_df.merge(src_df, left_on='id', right_on='PREDICATION_ID', how='inner')
print(joined_df.shape)
print(joined_df.columns)

# %% [markdown]
# **Impact of Argument distance on various performance matrices**

# %%
joined_df_verb=joined_df[joined_df["INDICATOR_TYPE"]=="VERB"]
print(joined_df_verb.shape)


# %%
#getting relevant columns
joined_df_verb= joined_df_verb[['id', 'y_true', 'y_pred', 'y_pred_0', 'y_pred_1','SUBJECT_DIST', 'SUBJECT_MAXDIST','OBJECT_DIST', 'OBJECT_MAXDIST']]
joined_df_verb.head()

# %%
import importlib
importlib.reload(utils)

# %%
# cumulative argument distance
joined_df_verb['DIST_SUM'] = joined_df_verb['SUBJECT_DIST'] + joined_df_verb['OBJECT_DIST']

# %%
utils.getCumArgDistImpact(df=joined_df_verb, name=f"{KEY}_cum_arg_dist_impact_verbal")

# %%
utils.getCatArgDistImpact(df=joined_df_verb, name=f"{KEY}_cat_arg_dist_impact_verbal")

# %%
utils.getPrecisionRecallCurve(df=joined_df_verb, name=f"{KEY}_precision_recall_curve_verbal")

# %%
utils.getSubObjHeatmap(df=joined_df_verb,name=f"{KEY}_sub_obj_heatmap_verbal")

# %%
joined_df['DIST_SUM'] = joined_df['SUBJECT_DIST'] + joined_df['OBJECT_DIST']
utils.getCumArgDistImpact(df=joined_df, name=f"{KEY}_cum_arg_dis_impact_all")

# %%
utils.getCatArgDistImpact(df=joined_df, name=f"{KEY}_cat_arg_dis_impact_all")

# %%
utils.getPrecisionRecallCurve(df=joined_df, name=f"{KEY}_precision_recall_curve_all")

# %%
utils.getSubObjHeatmap(df=joined_df, name=f"{KEY}_sub_obj_heatmap_all")


