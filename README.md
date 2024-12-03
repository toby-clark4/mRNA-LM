## Pre-trained models
The CodonBERT Pytorch model can be downloaded [here](https://cdn.prod.accelerator.sanofi/llm/CodonBERT.zip). The artifact is under a [license](ARTIFACT_LICENSE.md).
The code and repository are under a [software license](SOFTWARE_LICENSE.md).

## Finetune the mRNA-LM model 
```python finetune_all.py --task halflife ```
### command-line arguments:
- --task, -t     (str,   default=""):  **Required.** Task. Specify the task to be performed by the model.
- --output, -o   (str,   default=""):  **Required.** Output folder. Specify the output folder to save models. 
- --batch, -b    (int,   default=64):    Batch size. Specify the size of each batch.
- --lr           (float, default=1e-5):  Learning rate. Modify this to change the learning rate. The comment mentions 2e-5 as an alternative.
- --device, -d   (int,   default=0):     GPU device. Specify the GPU ID to run the model.
- --lorar        (int,   default=32):    Lora rank. Adjust this to control the rank of Lora.
- --lalpha       (int,   default=32):    Lora alpha. Adjust this to control the alpha parameter of Lora.
- --ldropout     (float, default=0.5):   Lora dropout. Adjust this to control the dropout rate of Lora.
- --head_dim     (int,   default=768):   Production head dimension. Modify this to change the dimension of the production head.
- --head_dropout (float, default=0.5):   Production head dropout. Modify this to change the dropout rate of the production head.
- --useCLIP      (bool,  default=False): Whether to use CLIP. Adjust this to turn the CLIP loss on or off.
- --temperature  (float, default=0.07):  Temperature parameter in CLIP loss. Modify this to change the temperature for the CLIP loss.
- --coefficient  (float, default=0.2):   Adjustment coefficient for CLIP loss. Modify this to change the coefficient for the CLIP loss.
