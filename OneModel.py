import torch
import torch.nn as nn
from transformers.activations import gelu

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import *
from tokenizers.processors import BertProcessing

from transformers import BertForSequenceClassification, PreTrainedTokenizerFast

########### PEFT
from peft import LoraConfig, TaskType
from peft import get_peft_model

class OneModel(torch.nn.Module):
    def __init__(self, region, num_labels, class_weights, lorar, lalpha, ldropout, output_hidden_states=False):
        super(OneModel, self).__init__()
        
        self.region = region 
        self.max_length = 1024
        if self.region == "5utr":
            self.max_length = 512
        
        # tokenizer
        self.tokenizer = None
        self.build_tokenizer()
        
        # model 
        if self.region == "5utr":
            model_dir = "/mount/data/models/mrna_5utr_model"
        elif self.region == "3utr":
            model_dir = "/mount/data/models/mrna_3utr_model"
        elif self.region == "cds":
            model_dir = "/mount/data/models/CodonBERT"
        else:
            print("wrong region!!", self.region)
            exit(0)
        
        self.model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels, output_hidden_states=output_hidden_states)
#         self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        ########### lora
        if lorar > 0:
            peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                                    r=lorar, 
                                    lora_alpha=lalpha, 
                                    lora_dropout=ldropout,
                                    use_rslora=True)

            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
#             self.model.gradient_checkpointing_enable()
#             self.model.enable_input_require_grads()
    
    def build_tokenizer(self):
        lst_ele = list('AUGCN')
        lst_voc = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        if self.region == "cds":
            for a1 in lst_ele:
                for a2 in lst_ele:
                    for a3 in lst_ele:
                        lst_voc.extend([f'{a1}{a2}{a3}'])
        else:
            for a1 in lst_ele:
                lst_voc.extend([f'{a1}'])
                        
        dic_voc = dict(zip(lst_voc, range(len(lst_voc))))
        tokenizer = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
        tokenizer.add_special_tokens(['[PAD]','[CLS]', '[UNK]', '[SEP]','[MASK]'])
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = BertProcessing(
            ("[SEP]", dic_voc['[SEP]']),
            ("[CLS]", dic_voc['[CLS]']),
        )

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, 
                                                 unk_token='[UNK]',
                                                 sep_token='[SEP]',
                                                 pad_token='[PAD]',
                                                 cls_token='[CLS]',
                                                 mask_token='[MASK]')

    def encode_string(self, data):
        return self.tokenizer(data[self.region], 
                              truncation=True,  # do_not_truncate
                              padding="max_length",
                              max_length=self.max_length)