import os
import argparse
import torch
import numpy as np

from transformers import  TrainingArguments, Trainer, EarlyStoppingCallback
from scipy.stats import pearsonr, spearmanr
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, f1_score

from FullModel import FullModel
from dataload import *

os.environ["TOKENIZERS_PARALLELISM"] = "true"

######### Arguments Processing
parser = argparse.ArgumentParser(description='FullModel')

parser.add_argument('--task',   '-t', required=True, type=str, default="", help='task')
parser.add_argument('--output', '-o', required=True, type=str, default="", help='output dir') 

parser.add_argument('--lorar',    type=int, default=32, help='Lora rank')
parser.add_argument('--lalpha',   type=int, default=32, help='Lora alpha')
parser.add_argument('--ldropout', type=int, default=0.5, help='Lora dropout')
parser.add_argument('--lr',       type=float, default=1e-5, help='learning rate')

parser.add_argument('--head_dim', type=int, default=768, help='production head dimension')
parser.add_argument('--head_dropout', type=float, default=0.5, help='production head dropout')

parser.add_argument('--device', '-d', type=int, default=0, help='device')
parser.add_argument('--batch',  '-b', type=int, default=64, help='batch size')

parser.add_argument('--useCLIP',      '-clip', type=bool, default=False, help='use CLIP')
parser.add_argument('--temperature',  '-temp', type=float, default=0.07, help='temperature')
parser.add_argument('--coefficient',  '-coeff', type=float, default=0.2, help='coefficient')

parser.add_argument('--splice', '-s', type=bool, default=False, help='use splice sites')

args = parser.parse_args()

########### GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.device

######### Task
if args.task not in ["tr", "halflife", "5class", "liver"]:
    print("Wrong task!", args.task)
    exit(0)

num_labels = 1
class_weights = []
metric_for_best_model = ""
greater_is_better = True
if args.task == "5class":
    num_labels = 5
    class_weights = [0.97326057, 0.48056585, 1.24829396, 1.44412955, 2.51197183]
    metric_for_best_model = "auroc"
else:
    metric_for_best_model = "spearmanr"

df_list = []
preds_list = []
true_list = []
for i in range(10):
    out_dir = f"{args.output}/fold_{i}"
    ########### loading pretrained model and downstream task model
    model = FullModel(num_labels, class_weights, 
                      args.lorar, args.lalpha, args.ldropout, 
                      args.head_dim, args.head_dropout,
                      args.useCLIP, args.temperature, args.coefficient, cdslm=True, splice=args.splice)
    
    ########### loading dataset and dataloader
    if args.task == "tr":
        ds_train, ds_valid, ds_test = build_dp_dataset()
    elif args.task == "halflife":
        ds_train, ds_valid, ds_test = build_saluki_dataset(i, False, args.splice)
    elif args.task == "5class":
        ds_train, ds_valid, ds_test = build_class_dataset()
    elif args.task == "liver":
        ds_train, ds_valid, ds_test = build_liver_dataset()
    
    
    train_loader = ds_train.map(model.encode_string, batched=True)
    val_loader = ds_valid.map(model.encode_string, batched=True)
    test_loader = ds_test.map(model.encode_string, batched=True)
    
    ######### Training Settings & Metrics 
    training_args = TrainingArguments(
        optim='adamw_torch',
        learning_rate=args.lr,                     # learning rate
        output_dir=out_dir,                    # output directory to where save model checkpoint
        eval_strategy="epoch",                     # evaluate each `logging_steps` steps
        overwrite_output_dir=True,      
        num_train_epochs=100,                      # number of training epochs, feel free to tweak
        per_device_train_batch_size=args.batch,    # the training batch size, put it as high as your GPU memory fits
        per_device_eval_batch_size=args.batch,     # evaluation batch size
        gradient_accumulation_steps=2,
        save_strategy="epoch", 
        save_steps=1,                              # save model 
        load_best_model_at_end=True,               # whether to load the best model (in terms of loss) at the end of training
    #     metric_for_best_model=metric_for_best_model, 
    #     greater_is_better=greater_is_better,
        save_total_limit = 1,
        eval_steps=1,      
        logging_steps=100,
        report_to=["wandb"],
        save_safetensors=False,
        bf16=True,
    )
    
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
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5, early_stopping_threshold=0.0005,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=val_loader,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    
    ######### Training & Evaluation & Prediction
    # Train the model
    trainer.train() # resume_from_checkpoint=True
    """
    checkpoint = torch.load("/home/jovyan/shared/toby/cds-lm/mRNA-LM/checkpoint-2106/pytorch_model.bin", weights_only=True)
    model.load_state_dict(checkpoint)
    trainer.model = model
    """
    # Evaluate the model
    metrics = trainer.evaluate()
    print(metrics)
    pd.DataFrame(metrics, index=[0]).to_csv(f'./results/clip/eval_res_fold_{i}.csv')
    # Prediction on test set
    pred, true, metrics = trainer.predict(test_loader)
    print(metrics)
    df_list.append(pd.DataFrame(metrics, index = [0]))
    preds_list.extend(pred)
    true_list.extend(true)

res_df = pd.concat(df_list, ignore_index=True)
print(res_df['test_spearmanr'].mean())
res_df.to_csv('./CDS-LM-X-test_res.csv')
pd.DataFrame({'pred': preds_list, 'y_true': true_list}).to_csv('./CDS-LM-X-test_preds.csv')