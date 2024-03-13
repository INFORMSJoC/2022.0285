import math
import sys
sys.path.append("../")
from model.modeling_bert_linear_wo_norm import BertPreTrainedModel, BertModel, TwoModalLayer,  get_extended_attention_mask
from torch import nn as nn
import torch
from torch.nn import CrossEntropyLoss
from transformers.file_utils import (
    ModelOutput
)
from dataclasses import dataclass
from typing import Optional
import copy
import re

@dataclass
class ClauseOutput(ModelOutput):
    """
    Base class for outputs of token classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    clause_loss: Optional[torch.FloatTensor] = None
    clause_logits: torch.FloatTensor = None
    input_embeddings: Optional[list] = None
    output_embeddings: Optional[list] = None

class BertClause(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config,co_hidden_size=768, co_intermediate_size=1536, clause_max_len=None, max_len=None):
        super().__init__(config)
        config.co_hidden_size = co_hidden_size
        config.co_intermediate_size = co_intermediate_size
        config.co_num_attention_heads = config.attention_head
        self.config = config
        self.clause_max_len = clause_max_len
        self.max_len = max_len
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.src_dropout = nn.Dropout(self.config.src_dropout_prob)
        multmodal_layer = TwoModalLayer(config)
        self.feat_dim = self.config.hidden_size

        # context encoder
        self.bert = BertModel(self.config, add_pooling_layer=False)

        # coattention layer
        self.r2cattention = nn.ModuleList(
            [copy.deepcopy(multmodal_layer) for _ in range(self.config.num_connection_layers)]
        )
        self.s2cattention = nn.ModuleList(
            [copy.deepcopy(multmodal_layer) for _ in range(self.config.num_connection_layers)]
        )
        self.c2sattention = nn.ModuleList(
            [copy.deepcopy(multmodal_layer) for _ in range(self.config.num_connection_layers)]
        )

        # classifier
        self.classifier = nn.Linear(self.config.hidden_size * 2, 2)

        self.init_weights()

    def prepare_dt_fixup(self, params, init_mode="normal"):
        temp_state_dict = {}
        for name, param in params:
            if re.search(r'(.*)(query|key|value)(.*)', name) and not re.search(
                    r'(.*)(LayerNorm|layernorm)(.*)', name):
                # print("1", name)
                if "bias" not in name:
                    if init_mode == "normal":
                        nn.init.normal_(param, std=self.config.initializer_range)
                    else:
                        nn.init.xavier_uniform_(param, gain=1 / math.sqrt(2))
                temp_state_dict[name] = param
            elif re.search(r'(.*)(intermediate|output)(.*)',
                           name) and not re.search(r'(.*)(LayerNorm|layernorm)(.*)', name):
                # print("2", name)
                if "bias" not in name:
                    if "bias" not in name:
                        if init_mode == "normal":
                            nn.init.normal_(param, std=self.config.initializer_range)
                        else:
                            nn.init.xavier_uniform_(param)
                temp_state_dict[name] = param

        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dict)

    def dt_fixup_initialization(self, params, max_norm=10.0):
        norm_factor = (3 * max_norm) ** 2
        factor = (norm_factor * self.config.num_connection_layers * 2) ** 0.5
        # print(scale)
        temp_state_dict = {}
        for name, param in params:
            if re.search(r'(.*)(value)(.*)', name) and not re.search(r'(.*)(LayerNorm|layernorm)(.*)', name):
                # print("1", name)
                temp_state_dict[name] = (param * (2 ** 0.5)) / factor
            # Note that classifier weight is also included in the re-scaling
            elif re.search(r'(.*)(intermediate|output)(.*)', name) and not re.search(r'(.*)(LayerNorm|layernorm)(.*)',
                                                                                     name):
                # print("2", name)
                temp_state_dict[name] = param / factor

        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dict)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, input_mask=None,
                clause_idx=None, sarcasm_idx=None, subreddit_idx=None,
                subreddit_input_ids=None, subreddit_attention_mask=None, subreddit_token_type_ids=None, subreddit_position_ids=None,
                clause_labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=return_dict
        )

        subreddit_outputs = self.bert(
            subreddit_input_ids,
            attention_mask=subreddit_attention_mask,
            token_type_ids=subreddit_token_type_ids,
            position_ids=subreddit_position_ids,
            return_dict=return_dict
        )


        bert_sequence_output = bert_outputs[0]
        subreddit_sequence_output = subreddit_outputs[0]
        input_embeddings = [bert_sequence_output, subreddit_sequence_output]
        bert_sequence_output = self.src_dropout(bert_sequence_output)
        subreddit_sequence_output = self.src_dropout(subreddit_sequence_output)
        input_bert_embeddings = bert_sequence_output
        extended_attention_mask = get_extended_attention_mask(attention_mask, attention_mask.shape)
        extended_subreddit_attention_mask = get_extended_attention_mask(subreddit_attention_mask,
                                                                        subreddit_attention_mask.shape)
        for i in range(self.config.num_connection_layers):
            output_bert_embeddings = self.r2cattention[i](input_bert_embeddings, extended_attention_mask,
                                                            subreddit_sequence_output, extended_subreddit_attention_mask)
            input_bert_embeddings = output_bert_embeddings
        sent_list = list(torch.split(output_bert_embeddings, self.clause_max_len, dim=1))
        sent_embeddings = torch.stack(sent_list, dim=1)
        clause_embeddings = self.sents_index_select(sent_embeddings, clause_idx)
        batch_size, clause_num = clause_embeddings.shape[0], clause_embeddings.shape[1]
        sarcasm_embeddings = self.sents_index_select(sent_embeddings, sarcasm_idx)

        sarcasm_embeddings = sarcasm_embeddings.expand(clause_embeddings.shape)

        assert clause_embeddings.shape == sarcasm_embeddings.shape

        clause_embeddings = clause_embeddings.contiguous().view(-1, clause_embeddings.shape[-2], clause_embeddings.shape[-1])
        sarcasm_embeddings = sarcasm_embeddings.contiguous().view(-1, sarcasm_embeddings.shape[-2], sarcasm_embeddings.shape[-1])
        sent_attention_mask = torch.split(attention_mask, self.clause_max_len,
                                     dim=1)  # att_mask [batch_size, clause_max_len] * max_sents_num

        sent_attention_mask = torch.stack(sent_attention_mask, dim=1)  # BERT att_mask [batch_size, max_sents_num, clause_max_len]
        clause_attention_mask = self.att_mask_index_select(sent_attention_mask,
                                                            clause_idx)
        sarcasm_attention_mask = self.att_mask_index_select(sent_attention_mask,
                                                  sarcasm_idx)  # sarcasm att_mask # [batch_size, 1, clause_max_len]
        sarcasm_attention_mask = sarcasm_attention_mask.expand(clause_attention_mask.shape)  # [batch_size, max_sents_num, clause_max_len]
        assert clause_attention_mask.shape == sarcasm_attention_mask.shape
        clause_attention_mask = clause_attention_mask.contiguous().view(-1, clause_attention_mask.shape[-1])
        sarcasm_attention_mask = sarcasm_attention_mask.contiguous().view(-1, sarcasm_attention_mask.shape[-1])

        extended_clause_attention_mask = get_extended_attention_mask(clause_attention_mask, clause_attention_mask.shape)
        extended_sarcasm_attention_mask = get_extended_attention_mask(sarcasm_attention_mask, sarcasm_attention_mask.shape)

        input_clause_embeddings = aux_clause_embeddings = clause_embeddings
        input_sarcasm_embeddings = aux_sarcasm_embeddings = sarcasm_embeddings
        for i in range(self.config.num_connection_layers):
            output_clause_embeddings = self.s2cattention[i](input_clause_embeddings, extended_clause_attention_mask,
                                                            aux_sarcasm_embeddings, extended_sarcasm_attention_mask)
            output_sarcasm_embeddings = self.c2sattention[i](input_sarcasm_embeddings, extended_sarcasm_attention_mask,
                                                             aux_clause_embeddings, extended_clause_attention_mask)
            input_clause_embeddings = output_clause_embeddings
            input_sarcasm_embeddings = output_sarcasm_embeddings

        clause_h = output_clause_embeddings[:, 0]
        sarcasm_h = output_sarcasm_embeddings[:, 0]
        con_feature = torch.cat([clause_h, sarcasm_h], dim=-1)
        clause_logits = self.classifier(self.dropout(con_feature))
        clause_logits = clause_logits.reshape(batch_size, clause_num, -1)
        output_embeddings = [torch.cat([output_clause_embeddings, output_sarcasm_embeddings], dim=1)]

        clause_loss = None
        if clause_labels is not None:

            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = clause_labels.view(-1) == -100
            active_logits = clause_logits.view(-1, 2)
            active_labels = torch.where(
                active_loss, torch.tensor(loss_fct.ignore_index).type_as(clause_labels), clause_labels.view(-1)
            )

            clause_loss = loss_fct(active_logits, active_labels)

        if not return_dict:
            output = (clause_logits,) + clause_outputs[2:]
            return ((clause_loss,) + output) if loss is not None else output

        return ClauseOutput(
            clause_loss=clause_loss,
            clause_logits=clause_logits,
            input_embeddings=input_embeddings if input_embeddings else None,
            output_embeddings=output_embeddings if output_embeddings else None
        )

    def batched_index_select(self, sequence_output, clause_idx):
        dummy = clause_idx.unsqueeze(2).expand(clause_idx.size(0), clause_idx.size(1), sequence_output.size(2))
        clause_h = sequence_output.gather(1, dummy)
        return clause_h

    def sents_index_select(self, sents_output, idxs):
        dummy = idxs.unsqueeze(2).unsqueeze(3).expand(idxs.size(0), idxs.size(1), sents_output.size(2), sents_output.size(3)) # [batch_size, 1, clause_max_len, 768]
        select_output = sents_output.gather(1, dummy) # [batch_size, 1, clause_max_len, 768]
        return select_output

    def att_mask_index_select(self, sents_mask, idxs):
        dummy = idxs.unsqueeze(2).expand(idxs.size(0), idxs.size(1), sents_mask.size(2)) # [batch_size, 1, clause_max_len]
        select_mask = sents_mask.gather(1, dummy) # [batch_size, 1, clause_max_len]
        return select_mask

