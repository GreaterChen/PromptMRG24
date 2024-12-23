import os
import warnings
warnings.filterwarnings("ignore")

from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
from models.resnet import blip_resnet

import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import Transformer

CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]

class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 args,
                 tokenizer=None,
                 image_size = 224,
                 prompt = '',
                 ):
        super().__init__()
        self.args = args
        
        vision_width = 2048
        self.visual_encoder = blip_resnet(args)
        
        self.cls_head = nn.Linear(vision_width+512, 18*4)
        nn.init.normal_(self.cls_head.weight, std=0.001)
        if self.cls_head.bias is not None:
            nn.init.constant_(self.cls_head.bias, 0)

        self.vision_proj = nn.Linear(vision_width, 512)

        self.tokenizer = tokenizer   
        
        decoder_config = BertConfig.from_json_file('/home/chenlb/xray_compare_model/PromptMRG/configs/bert_config.json')
        decoder_config.encoder_width = vision_width
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)
        
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        self.memory = Transformer(d_model=512,
                                  num_encoder_layers=2,
                                  num_decoder_layers=2,
                                  num_queries=1)
        
    def forward(self, image, caption, cls_labels, clip_memory, criterion_cls, base_probs):
        image_embeds, avg_embeds = self.visual_encoder(image)   # (bs, 7*7, 2048), (bs, 2048)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        loss_cls = torch.tensor(0., dtype=torch.float32)
        caption = [item[108:] for item in caption]

        text = self.tokenizer(caption, padding='longest', truncation=True, return_tensors="pt").to(image.device)
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        # 在 PyTorch 的 CrossEntropyLoss 中，-100 是一个特殊的值，用于表示忽略某些位置的损失计算
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:,:self.prompt_length] = -100

        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
          
        loss_lm = decoder_output.loss                
        return loss_lm, loss_cls
        
def generate(self, image, clip_memory, sample=False, num_beams=3, max_length=100, min_length=10, top_p=0.9, repetition_penalty=1.0):
    # Get image embeddings from visual encoder
    image_embeds, _ = self.visual_encoder(image) 
    
    if not sample:
        image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        
    # Create attention mask for image embeddings
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
    model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
    
    # Initialize decoder input with BOS token
    input_ids = torch.ones((image_embeds.size(0), 1), dtype=torch.long).to(image.device)
    input_ids[:, 0] = self.tokenizer.bos_token_id
    
    # Generate text using beam search
    outputs = self.text_decoder.generate(input_ids=input_ids,
                                       min_length=min_length,
                                       max_new_tokens=max_length, 
                                       num_beams=num_beams,
                                       eos_token_id=self.tokenizer.sep_token_id,
                                       pad_token_id=self.tokenizer.pad_token_id,
                                       repetition_penalty=repetition_penalty,
                                       **model_kwargs)
    
    # Decode generated tokens to text
    captions = []
    for output in outputs:
        caption = self.tokenizer.decode(output, skip_special_tokens=True)
        captions.append(caption)
        
    return captions

def blip_decoder(args, tokenizer, **kwargs):
    model = BLIP_Decoder(args, tokenizer, **kwargs)
    return model    
    
