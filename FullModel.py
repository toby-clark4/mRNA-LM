import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import gelu
from torch.nn.functional import softmax, log_softmax

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import *
from tokenizers.processors import BertProcessing

from transformers import BertForMaskedLM, PreTrainedTokenizerFast

########### PEFT
from peft import LoraConfig, TaskType
from peft import get_peft_model

class FullModel(torch.nn.Module):
    def __init__(self, num_labels, class_weights, lorar, lalpha, ldropout, head_dim=768, head_droupout=0.5, useCLIP=False, temperature=0.07, clip_coeff=0.2):
        super(FullModel, self).__init__()
        
        # tokenizer
        self.tokenizer_cds = None
        self.tokenizer_5utr = None
        self.tokenizer_3utr = None
        self.build_tokenizer()
        self.CLIP = useCLIP
        
        # model 
        self.utr5 = BertForMaskedLM.from_pretrained("/mount/data/models/mrna_5utr_model")
        self.utr3 = BertForMaskedLM.from_pretrained("/mount/data/models/mrna_3utr_model")
        self.cds = BertForMaskedLM.from_pretrained("/mount/data/models/CodonBERT")
        
        # gradient_checkpointing_enable: trading speed for memory
        # self.utr5.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # self.utr3.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # self.cds.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        ########### lora
        if lorar > 0:
            peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS,
                                    r=lorar, 
                                    lora_alpha=lalpha, 
                                    lora_dropout=ldropout,
                                    use_rslora=True)

            self.utr5 = get_peft_model(self.utr5, peft_config)
            self.utr5.print_trainable_parameters()
            # self.utr5.gradient_checkpointing_enable()
            # self.utr5.enable_input_require_grads()

            self.utr3 = get_peft_model(self.utr3, peft_config)
            self.utr3.print_trainable_parameters()
            # self.utr3.gradient_checkpointing_enable()
            # self.utr3.enable_input_require_grads()

            self.cds = get_peft_model(self.cds, peft_config)
            self.cds.print_trainable_parameters()
            # self.cds.gradient_checkpointing_enable()
            # self.cds.enable_input_require_grads()
            

        # Dense layers for CLIP-style structure
        self.dense_utr5 = nn.Linear(768, 768)
        self.dense_cds1 = nn.Linear(768, 768)
        self.dense_cds2 = nn.Linear(768, 768)
        self.dense_utr3 = nn.Linear(768, 768)

        self.final_dense = nn.Linear(768*3, head_dim)

        self.transform_act_fn = gelu
        self.LayerNorm = torch.nn.LayerNorm(head_dim, eps=1e-12)
        self.dropout = nn.Dropout(head_droupout)

        self.decoder = nn.Linear(head_dim, num_labels, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_labels))
        self.decoder.bias = self.bias
        
        if num_labels == 1:
            self.loss_fn = nn.MSELoss()
        else:
            class_weights=torch.tensor(class_weights, dtype=torch.float)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')

        # Temperature for scaling logits
        self.temperature = temperature
        self.clip_coeff = clip_coeff
        self.is_first_epoch = True
        

    def cross_entropy_loss(self, preds, targets, reduction='none'):
        log_softmax_preds = log_softmax(preds, dim=-1)
        loss = (-targets * log_softmax_preds).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        
    def contrastive_loss(self, embeds1, embeds2):
        # Normalize the embeddings
        embeds1 = nn.functional.normalize(embeds1, p=2, dim=1)
        embeds2 = nn.functional.normalize(embeds2, p=2, dim=1)
        
        # Calculate similarity matrix
        logits = torch.matmul(embeds1, embeds2.t()) / self.temperature
        similarity_1 = torch.matmul(embeds1, embeds1.t())
        similarity_2 = torch.matmul(embeds2, embeds2.t())
        
        # Calculate targets
        targets = softmax((similarity_1 + similarity_2) / 2 * self.temperature, dim=-1)
        
        # Calculate cross-entropy loss
        loss1 = self.cross_entropy_loss(logits, targets, reduction='none')
        loss2 = self.cross_entropy_loss(logits.t(), targets.t(), reduction='none')
        
        return (loss1.mean() + loss2.mean()) / 2
    
    def combine_embeds(self, input_ids, attention_mask, model, model_max_seq_length):
        # maximum length in the batch
        seq_len = torch.sum(attention_mask, 1)
        max_seq_length = torch.max(seq_len).item()

        i = 0 
        embeds = []
        while i < max_seq_length:
            features = {"input_ids": input_ids[:, i:min(max_seq_length, i+model_max_seq_length-2)], "attention_mask": attention_mask[:, i:min(max_seq_length, i+model_max_seq_length-2)]}
            output_states = model(**features, output_hidden_states=True)
            embeds.append(output_states["hidden_states"][-1])
            i += model_max_seq_length - 2
        embeds = torch.cat(embeds, 1)
        # print(embeds.size())

        return embeds, attention_mask[:, :max_seq_length]

    def get_mean_token_embeddings(self, token_embeddings, token_mask):
        input_mask_expanded = token_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / sum_mask

        return sum_embeddings


    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, input_ids3, attention_mask3, labels, return_hidden=False, epoch=None, decay_rate=0.95, **kwargs):
        utr5_embeds = self.utr5(input_ids=input_ids1, attention_mask=attention_mask1, output_hidden_states=True)["hidden_states"][-1]
        cds_embeds  = self.cds(input_ids=input_ids2, attention_mask=attention_mask2, output_hidden_states=True)["hidden_states"][-1]
        utr3_embeds = self.utr3(input_ids=input_ids3, attention_mask=attention_mask3, output_hidden_states=True)["hidden_states"][-1]

        utr5_sum_embeddings = self.get_mean_token_embeddings(utr5_embeds[:, 1:-1, :], attention_mask1[:, 1:-1])
        cds_sum_embeddings  = self.get_mean_token_embeddings(cds_embeds[:, 1:-1, :], attention_mask2[:, 1:-1])
        utr3_sum_embeddings = self.get_mean_token_embeddings(utr3_embeds[:, 1:-1, :], attention_mask3[:, 1:-1])

        if not self.CLIP:
            joint_embed = torch.cat([utr5_sum_embeddings, cds_sum_embeddings, utr3_sum_embeddings], dim=1)

            hidden_states = self.final_dense(joint_embed)
            hidden_states = self.transform_act_fn(hidden_states)
            hidden_states = self.LayerNorm(hidden_states)

            hidden_states = self.dropout(hidden_states)
            logits = self.decoder(hidden_states).squeeze()
            loss = self.loss_fn(logits, labels)

            if not return_hidden:
                return loss, logits

            return joint_embed, hidden_states

        # CLIP-style transformations
        utr5_transformed = self.dense_utr5(utr5_sum_embeddings)
        cds_transformed1 = self.dense_cds1(cds_sum_embeddings)
        cds_transformed2 = self.dense_cds2(cds_sum_embeddings)
        utr3_transformed = self.dense_utr3(utr3_sum_embeddings)

        # Apply CLIP-style contrastive loss with no_grad to avoid affecting the main graph
        clip_loss1 = self.contrastive_loss(utr5_transformed, cds_transformed1)
        clip_loss2 = self.contrastive_loss(cds_transformed2, utr3_transformed)
        average_clip_loss = (clip_loss1 + clip_loss2) / 2

        # Combine the embeddings for the final classification task
        combined_hidden_states = torch.cat([utr5_transformed, cds_transformed1, utr3_transformed], dim=1)

        combined_hidden_states = self.final_dense(combined_hidden_states)
        combined_hidden_states = self.transform_act_fn(combined_hidden_states)
        combined_hidden_states = self.LayerNorm(combined_hidden_states)

        combined_hidden_states = self.dropout(combined_hidden_states)
        logits = self.decoder(combined_hidden_states).squeeze()
        classification_loss = self.loss_fn(logits, labels) 

        # Initialize clip_coeff if first epoch
        if self.is_first_epoch:
            self.is_first_epoch = False
            self.clip_coeff = classification_loss.item() / average_clip_loss.item() * self.clip_coeff

        # Total loss
        total_loss = classification_loss + average_clip_loss * self.clip_coeff

        return total_loss, logits


    def compute_contrastive_loss(self, utr5_proj, cds_proj, utr3_proj, temperature=0.07):
        # Normalize the projections
        utr5_proj_norm = F.normalize(utr5_proj, dim=-1)
        cds_proj_norm = F.normalize(cds_proj, dim=-1)
        utr3_proj_norm = F.normalize(utr3_proj, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(utr5_proj_norm, cds_proj_norm.T) / temperature
        
        # Labels for contrastive loss
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        
        # Compute contrastive loss
        loss_fct = nn.CrossEntropyLoss()
        contrastive_loss = loss_fct(similarity_matrix, labels) + loss_fct(similarity_matrix.T, labels)
        
        return contrastive_loss
    

    def build_tokenizer(self):
        lst_ele = list('AUGCN')
        lst_voc = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        for a1 in lst_ele:
            for a2 in lst_ele:
                for a3 in lst_ele:
                    lst_voc.extend([f'{a1}{a2}{a3}'])
        dic_voc = dict(zip(lst_voc, range(len(lst_voc))))
        tokenizer_cds = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
        tokenizer_cds.add_special_tokens(['[PAD]','[CLS]', '[UNK]', '[SEP]','[MASK]'])
        tokenizer_cds.pre_tokenizer = Whitespace()
        tokenizer_cds.post_processor = BertProcessing(
            ("[SEP]", dic_voc['[SEP]']),
            ("[CLS]", dic_voc['[CLS]']),
        )
        # tokenizer_5utr
        lst_voc = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        for a1 in lst_ele:
            lst_voc.extend([f'{a1}'])
        dic_voc = dict(zip(lst_voc, range(len(lst_voc))))
        tokenizer_5utr = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
        tokenizer_5utr.add_special_tokens(['[PAD]','[CLS]', '[UNK]', '[SEP]','[MASK]'])
        tokenizer_5utr.pre_tokenizer = Whitespace()
        tokenizer_5utr.post_processor = BertProcessing(
            ("[SEP]", dic_voc['[SEP]']),
            ("[CLS]", dic_voc['[CLS]']),
        )
        tokenizer_3utr = tokenizer_5utr

        self.tokenizer_cds = PreTrainedTokenizerFast(tokenizer_object=tokenizer_cds, 
                                                     unk_token='[UNK]',
                                                     sep_token='[SEP]',
                                                     pad_token='[PAD]',
                                                     cls_token='[CLS]',
                                                     mask_token='[MASK]')
        self.tokenizer_5utr = PreTrainedTokenizerFast(tokenizer_object=tokenizer_5utr, 
                                                      unk_token='[UNK]',
                                                      sep_token='[SEP]',
                                                      pad_token='[PAD]',
                                                      cls_token='[CLS]',
                                                      mask_token='[MASK]')
        self.tokenizer_3utr = PreTrainedTokenizerFast(tokenizer_object=tokenizer_3utr, 
                                                      unk_token='[UNK]',
                                                      sep_token='[SEP]',
                                                      pad_token='[PAD]',
                                                      cls_token='[CLS]',
                                                      mask_token='[MASK]')

    def encode_string(self, data):
        tok_5utr = self.tokenizer_5utr(data['5utr'], 
                                      truncation=True,  # do_not_truncate
                                      padding="max_length",
                                      max_length=512)
        tok_cds = self.tokenizer_cds(data['cds'], 
                                    truncation=True,  # do_not_truncate
                                    padding="max_length",
                                    max_length=1024)
        tok_3utr = self.tokenizer_3utr(data['3utr'], 
                                      truncation=True,  # do_not_truncate
                                      padding="max_length",
                                      max_length=1024)

        return {
            'input_ids1': tok_5utr['input_ids'],
            'attention_mask1': tok_5utr['attention_mask'],
            'input_ids2': tok_cds['input_ids'],
            'attention_mask2': tok_cds['attention_mask'],
            'input_ids3': tok_3utr['input_ids'],
            'attention_mask3': tok_3utr['attention_mask']
        }
