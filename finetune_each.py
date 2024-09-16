import os
import numpy as np
import pandas as pd
import argparse

import evaluate
from datasets import Dataset, interleave_datasets, concatenate_datasets

from transformers import TrainingArguments, Trainer

from scipy.stats import pearsonr, spearmanr
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, f1_score

import wandb

from dataload import *
from OneModel import OneModel

# from safetensors.torch import load_model, load_file
# from scipy.stats import zscore

########### PEFT
from peft import LoraConfig, TaskType
from peft import get_peft_model

######### Arguments Processing
parser = argparse.ArgumentParser(description='FullModel')

parser.add_argument('--lorar',    type=int, default=32, help='Lora rank')
parser.add_argument('--lalpha',   type=int, default=32, help='Lora alpha')
parser.add_argument('--ldropout', type=int, default=0.5, help='Lora dropout')
parser.add_argument('--lr',       type=float, default=1e-5, help='learning rate') # 2e-5

parser.add_argument('--device', '-d', type=int, default=1, help='device')
parser.add_argument('--batch',  '-b', type=int, default=128, help='batch size')

parser.add_argument('--cross',  '-c', type=int, default=0, help='batch size')
parser.add_argument('--region', '-r', type=str, default="", help='batch size')
parser.add_argument('--task',   '-t', type=str, default="bp", help='batch size')
parser.add_argument('--eval',         type=int, default=0, help='eval set')
parser.add_argument('--test',         type=int, default=1, help='test set')

args = parser.parse_args()

########### GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.device 
os.environ["TOKENIZERS_PARALLELISM"] = "true"

########### Task
if args.region not in ["5utr", "cds", "3utr"]:
    print("Wrong region", args.region)
    exit(0)

if args.task not in ["bp", "saluki_human_f0c0", "class", "liver", "ngs"]:
    print("Wrong task!", args.task)
    exit(0)
    
num_labels = 1
class_weights = []
if args.task == "class":
    num_labels = 5
    class_weights = [0.97326057, 0.48056585, 1.24829396, 1.44412955, 2.51197183]

output_dir = "%s_model_%s_eval%d_test%d" % (args.region, args.task, args.eval, args.test)
if os.path.exists(output_dir):
    resume = True
else:
    resume = False
    
######### wandb
wandb.init(
    project="full_mRNA_study_benchmarks",
    name="%s_%s" % (args.task, args.region),
    mode="disabled",
    config={
        "lorar": args.lorar,
        "lalpah": args.lalpha,
        "ldropout": args.ldropout,
        "region": args.region
    }
)

########### loading dataset
if args.task == "bp":
    ds_train, ds_val, ds_test = build_dp_dataset()
elif args.task == "saluki_human_f0c0":
    ds_train, ds_val, ds_test = build_saluki_dataset(0)
elif args.task == "class":
    ds_train, ds_val, ds_test = build_class_dataset()
elif args.task == "liver":
    ds_train, ds_val, ds_test = build_liver_dataset()
elif args.task == "ngs":
    if args.region == "cds":
        ds_train, ds_val, ds_test = build_ngs_dataset(args.eval, args.test)
    else:
        ds_train, ds_val, ds_test = build_ngs_dataset2(args.eval, args.test)
else:
    exit(0)

########### loading pretrained model and downstream task model
themodel = OneModel(args.region, num_labels, class_weights, args.lorar, args.lalpha, args.ldropout)
model = themodel.model

########### Tokenize dataset
train_loader = ds_train.map(themodel.encode_string, batched=True)
val_loader = ds_val.map(themodel.encode_string, batched=True)
test_loader = ds_test.map(themodel.encode_string, batched=True)

######### Training Settings & Metrics 
training_args = TrainingArguments(
    optim='adamw_torch',
    learning_rate=args.lr,                     # learning rate
    output_dir=output_dir,                     # output directory to where save model checkpoint
    eval_strategy="epoch",               # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=100,                      # number of training epochs, feel free to tweak
    per_device_train_batch_size=args.batch,    # the training batch size, put it as high as your GPU memory fits
    per_device_eval_batch_size=args.batch,     # evaluation batch size
    save_strategy="epoch", 
    save_steps=1,                               # save model 
    load_best_model_at_end=True,                # whether to load the best model (in terms of loss) at the end of training
    save_total_limit = 1,
    eval_steps=1,      
    logging_steps=1,
    report_to= "wandb",
    save_safetensors=False
)


metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    if num_labels == 1:
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


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader,
    eval_dataset=val_loader,
    compute_metrics=compute_metrics
)


######### Training & Evaluation & Prediction
# Train the model
trainer.train(resume_from_checkpoint=resume) # resume_from_checkpoint=True

# Evaluate the model
# print('>>>>> task: %s lr: %f freeze: %d' % (task_name.replace(" ", "-"), lr, args.freeze))
metrics = trainer.evaluate()
print(metrics)

# Prediction on test set
pred, _, metrics = trainer.predict(test_loader)
print(metrics)
