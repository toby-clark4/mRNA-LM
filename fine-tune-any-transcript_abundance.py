### Fine-tuning the pretrained MLM model for regression with cross-validation ###

import os
import time
import math
from datetime import datetime
import warnings
import numpy as np
import torch
import torch.nn as nn
import evaluate
import pandas as pd
from transformers import AlbertModel, BertModel, PreTrainedTokenizerFast, BertConfig, BertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_from_disk, Dataset
from sklearn.model_selection import KFold, PredefinedSplit, train_test_split
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from typing import Tuple
from scipy.stats import pearsonr, spearmanr
from torch.optim.lr_scheduler import LambdaLR
from optimi import StableAdamW
import wandb
from OneModel import OneModel

# Suppressing some useless warning. May not be ideal!
warnings.filterwarnings("ignore", category=FutureWarning)

print("-" * 53)
print("\tFINE-TUNING FOR REGRESSION")
print("-" * 53)

# Can safely ingore these functions - should just work
class TrainValTestSplit:
    """
    Adapter to extend cross-validators for train/val/test splits.
    Mimics scikit-learn's cross-validator interface.
    If test_fold is specified, use a predefined split like PredefinedSplit in scikit-learn
    """
    
    def __init__(self, n_folds=5, test_fold=None):
        """
        Parameters:
        test_fold: array-like, fold assignments (0-9 for 10-fold CV)
        """
        if test_fold:
            self.test_fold = np.array(test_fold)
            self.unique_folds = np.unique(self.test_fold)
            self.n_folds=None
        elif n_folds:
            self.n_folds=n_folds
        else:
            raise ValueError("Need to specify one of n_folds or a predefined test fold")
    
    def split(self, data=None):
        """
        Generate train/val/test splits.
        Returns (train_idx, val_idx, test_idx) tuples.
        """
        if self.n_folds:
            if data is None:
                raise ValueError("Need to provide data to split")
                
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_test_idx in kf.split(data):
                test_idx, val_idx = train_test_split(val_test_idx, test_size=0.5, shuffle=True, random_state=42)
                yield train_idx, val_idx, test_idx
        else:        
            for test_fold_val in self.unique_folds:
                val_fold_val = (test_fold_val + 1) % len(self.unique_folds)
                
                test_idx = np.where(self.test_fold == test_fold_val)[0]
                val_idx = np.where(self.test_fold == val_fold_val)[0]
                train_idx = np.where(~np.isin(self.test_fold, [test_fold_val, val_fold_val]))[0]
                
                yield train_idx, val_idx, test_idx

def tokenize_function(examples):
    return tokenizer(
        examples["codon_sequence"],
        truncation=True,
        padding="max_length",
        max_length=2048,
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if config.num_labels == 1:
        logits = logits.flatten()
        labels = labels.flatten()

        try:
            pearson_corr = pearsonr(logits, labels)[0].item()
            spearman_corr = spearmanr(logits, labels)[0].item()
            return {
                "pearson": pearson_corr,
                "spearmanr": spearman_corr,
            }
        except:
            return {"pearson":0.0, "spearmanr":0.0}
    else:
        predictions = np.argmax(logits, axis=-1)
        logits = softmax(logits, axis=1)
        
        f1 = f1_score(predictions, labels, average="macro")
        auroc = roc_auc_score(labels, logits, average="macro", multi_class='ovr')
        
        return {"f1": f1, "auroc": auroc}

def get_predictions(model, data) -> Tuple[float, float]:
    predictions = []
    labels = []

    with torch.no_grad():
        for example in data:
            # Move inputs to the device
            input_ids = torch.tensor([example["input_ids"]]).to(device)
            attention_mask = torch.tensor([example["attention_mask"]]).to(device)

            labels.append(example["labels"])
            # Forward pass to get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = outputs.logits.squeeze().detach().cpu().numpy()  # Move to CPU

            # Append prediction
            predictions.append(prediction)

    r, _ = pearsonr(predictions, labels)
    rho, _ = spearmanr(predictions, labels)

    return r, rho


def get_split_iterator(predefined_split: bool, data: pd.DataFrame, tokenized_data):
    """
    Returns an iterator for cross validation splitting
    """
    if predefined_split:
        split = PredefinedSplit(test_fold=data["split"])
        iterator = enumerate(split.split())
    else:
        indices = np.arange(len(tokenized_data))
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        iterator = enumerate(kf.split(indices))
    return iterator


def wsd_schedule(warmup_steps, stable_steps, total_steps, min_lr_ratio=0.05):
    """
    Warmup-Stable-Decay LR schedule with 1 - sqrt decay.

    Args:
        warmup_steps: Number of steps for linear warmup.
        stable_steps: Number of steps to keep LR constant after warmup.
        total_steps: Total number of training steps.
        min_lr_ratio: Minimum learning rate as a fraction of peak LR.
    """
    decay_steps = total_steps - warmup_steps - stable_steps

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps + stable_steps:
            return 1.0
        else:
            decay_progress = float(current_step - warmup_steps - stable_steps) / float(max(1, decay_steps))
            decay = 1.0 - math.sqrt(decay_progress)
            return max(min_lr_ratio, decay)

    return lr_lambda

# Change from here down
name = "CodonBERT"
pretrained_model_path = "/home/jovyan/workspace/mRNA-LM/models/codonbert"
config = BertConfig.from_pretrained(pretrained_model_path)

# Set paths and run configs
task = "transcript_abundance"
sequence_column = "sequence"
target_column = "logtpm"
freeze_layers = None # CHANGE HERE TO None if doing LoRA
lora = True # CHANGE THIS TO ACTIVATE LORA
k = 5 # How many folds
debug_mode = False  # True: If work on a small subset; 500 instances; False: for final fine-tuning
train_batch_size = 8 # 64
data_path = f"../CDS-LM/data/finetuning/transcript_abundance"# {task}"
res_dir = f"/home/jovyan/shared/toby/cds-lm/results/finetuning/{task}/"

config.problem_type = 'regression'
config.num_labels = 1
config.classifier_dropout = 0.3

# LoRA r --> larger with small dataset, smaller with large
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, # TOKEN_CLS
    r=8,
    lora_alpha=16,  # 2x sqrt(hidden size)
    lora_dropout=0.5,
    target_modules = ["query", "key", "value", "attention.output.dense"],
    modules_to_save=["classifier"],
    use_rslora=True,
)


today_date = datetime.today().strftime("%Y-%m-%d")

df_list = []
for species in ['athaliana', 'dmelanogaster', 'hsapiens', 'ppastoris', 'scerevisiae']:

    model_path = (
        f"/home/jovyan/shared/toby/cds-lm/assets/finetuning/{task}_{species}/{name}"
    )
    tokenized_name = f'{species}_codonbert'
    file_name = f"{species}.csv"
    checkpoint_dir = f"/home/jovyan/shared/toby/cds-lm/assets/checkpoints/finetuning/{name}/{task}_{species}"
    log_dir = f"./logs/finetuning/{task}/{name}"
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    tokenized_data_path = f"{data_path}/{tokenized_name}"
    
    print("\nRun config:")
    print(f"\tDate        : {today_date}")
    print(f"\tData Path   : {data_path}")
    print(f"\tModel Path  : {pretrained_model_path}")
    print(f"\tDebug Mode  : {debug_mode}")
    
    data = pd.read_csv(f"{data_path}/{file_name}", index_col=0)
    
    num_labels = 1
    class_weights = []
    themodel = OneModel("cds", num_labels, class_weights, 0, 0, 0)
    
    def mytok(seq, kmer_len, s, U=True):
        seq = seq.upper().replace("T", "U") if U else seq    
        kmer_list = []
        for j in range(0, (len(seq)-kmer_len)+1, s):
            kmer_list.append(seq[j:j+kmer_len])
        return kmer_list
    
    
    if os.path.exists(tokenized_data_path):
        print("\nTokenized dataset exists; loading it directly.")
        tokenized_data = load_from_disk(tokenized_data_path)
    else:
        print("\nTokenizing the datasets...")
        cds = [" ".join(mytok(seq, 3, 3)) for seq in data[sequence_column]]
        ds = Dataset.from_dict({'cds': cds, target_column: data[target_column]})
        tokenized_data = ds.map(themodel.encode_string, batched=True)
        tokenized_data = tokenized_data.rename_column(target_column, "labels")
    
        tokenized_data.save_to_disk(tokenized_data_path)
        print("Tokenized dataset saved to disk.")
    
    
    
    # iterator = get_split_iterator(predefined_split, data, tokenized_data)
    iterator = enumerate(TrainValTestSplit(n_folds=5).split(data))
    
    wandb.login(key=os.environ.get("WANDB_LOGIN"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    r_list = []
    rho_list = []
    # Iterate over folds
    # for fold, (train_idx, val_idx) in iterator:
    for fold, (train_idx, val_idx, test_idx) in iterator:
            
        run = wandb.init(
            entity="toby-clark",
            # Set the wandb project where this run will be logged.
            project="NVIDIA-Saturn",
            group=name,
        )
    
        
        # Train, validation and test splits for the current fold
        tokenized_train = tokenized_data.select(train_idx.tolist())
        tokenized_eval = tokenized_data.select(val_idx.tolist())
        tokenized_test = tokenized_data.select(test_idx.tolist())
        
        if debug_mode:
            tokenized_train = tokenized_train.select(range(50))
            
        print(f"Fold {fold + 1}")
    
        model = BertForSequenceClassification.from_pretrained(
                    pretrained_model_path,
                    config=config,
                )
    
    
        if lora:
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    
        steps_per_epoch = math.ceil(len(tokenized_train) / train_batch_size)
        eval_every = steps_per_epoch // 2
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f"{checkpoint_dir}/fold_{fold + 1}",
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_every,
            save_steps=eval_every,
            save_total_limit=2,
            learning_rate=5e-5, #5e-5 got best result
            warmup_ratio=0,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=train_batch_size,
            num_train_epochs=15,
            weight_decay=1e-3,
            # warmup_steps=400,
            gradient_accumulation_steps=1,
            logging_dir=f"{log_dir}/fold_{fold + 1}",
            logging_steps=100,
            load_best_model_at_end=True,
            optim='adamw_torch', # comment this out if using StableAdamW
            # max_grad_norm = 1.0, # comment this out if using StableAdamW
            bf16=True,
            seed=42,
            disable_tqdm=False,
            report_to=["wandb"],
        )
    
        # EarlyStoppingCallback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3, early_stopping_threshold=0.0001,
        )
        # Initialize the Trainer with early stopping callback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            callbacks=[early_stopping],
            compute_metrics=compute_metrics,
        )
    
        trainer.train()
        best_ckpt_path = trainer.state.best_model_checkpoint
    
        metrics = trainer.evaluate()
        print(metrics)
        
        # Prediction on test set
        pred, _, metrics = trainer.predict(tokenized_test)
        print(metrics)
        r_list.append(metrics['test_pearson'])
        rho_list.append(metrics['test_spearmanr'])
    
        # Save the fine-tuned model
        model.cpu()
        fold_model_path = f"{model_path}/fold_{fold + 1}"
        os.makedirs(fold_model_path, exist_ok=True)
        if lora:
            model = model.merge_and_unload()
        model.save_pretrained(fold_model_path)
    
        # Corresponding model loading script (KEEP THE NEXT LINE COMMENTED!)
        # model = CDSLMForSequenceClassification.from_pretrained(fold_model_path, config=config)
        
    res = pd.DataFrame({'r': r_list, 'rho': rho_list})
    res['Species'] = species
    df_list.append(res)
    print(f'Species: {species}')
    print(f'R: {np.mean(r_list):.4f}')
    print(f'rho: {np.mean(rho_list):.4f}')

pd.concat(df_list).to_csv(f'{res_dir}/{name}_test.csv')
"""
iterator = get_split_iterator(predefined_split, data, tokenized_data)  
r_list = []
rho_list = []

for fold, (train_index, test_index) in iterator:
    tokenized_eval = tokenized_data.select(test_index.tolist())
    ft_model_path = f"{model_path}/fold_{fold + 1}"
    model = ModernBertForSequenceClassification.from_pretrained(ft_model_path, config=config)
    model.to(device)
    model.eval()
    r, rho = get_predictions(model, tokenized_eval)
    r_list.append(r)
    rho_list.append(rho)
"""
