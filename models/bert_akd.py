"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import six

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss
from transformers.modeling_bert import BertModel, BertPreTrainedModel


class QAModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.teacher_model = None

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        query_mask=None,
        ans_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_position=None,
        end_position=None,
        output_attentions=None,
        output_hidden_states=None,
        adv_training=False
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        if start_position is not None and end_position is not None:
            kld_loss_func = KLDivLoss(reduction='none')
            softmax = nn.Softmax(dim=-1)
            log_softmax = nn.LogSoftmax(dim=-1)
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_start_logits, teacher_end_logits, _ = self.teacher_model(
                                                                    input_ids=input_ids,
                                                                    attention_mask=attention_mask,
                                                                    query_mask=query_mask,
                                                                    ans_mask=ans_mask,
                                                                    token_type_ids=token_type_ids,
                                                                    position_ids=position_ids,
                                                                    head_mask=head_mask,
                                                                    inputs_embeds=inputs_embeds,
                                                                    start_position=start_position,
                                                                    end_position=end_position,
                                                                    output_attentions=output_attentions,
                                                                    output_hidden_states=output_hidden_states,
                                                                )
        temp = 32.0

        if adv_training:
            inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
            inputs_embeds = inputs_embeds.clone().detach()
            start_position = start_position.clone().detach()
            end_position = end_position.clone().detach()
            teacher_start_logits = teacher_start_logits.clone().detach()
            teacher_end_logits = teacher_end_logits.clone().detach()
            inputs_embeds.requires_grad = True
            outputs = self.bert(
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            # [batch_size, seq_len, hidden_size]
            sequence_output = outputs[0]
            # [batch_size, seq_len, 2]
            logits = self.classifier(sequence_output)
            # [batch_size, seq_len, 1]
            start_logits, end_logits = logits.split(1, dim=-1)
            # [batch_size, seq_len]
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            # If we are on multi-GPU, split add a dimension
            if len(start_position.size()) > 1:
                start_position = start_position.squeeze(-1)
            if len(end_position.size()) > 1:
                end_position = end_position.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_position.clamp_(0, ignored_index)
            end_position.clamp_(0, ignored_index)

            # ce loss
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_position)
            end_loss = loss_fct(end_logits, end_position)
            ce_loss = (start_loss + end_loss) / 2

            # kld loss
            start_kld_loss = kld_loss_func(log_softmax(start_logits), softmax(teacher_start_logits / temp))
            kld_loss = torch.mean(torch.sum(start_kld_loss, dim=1))
            end_kld_loss = kld_loss_func(log_softmax(end_logits), softmax(teacher_end_logits / temp))
            kld_loss += torch.mean(torch.sum(end_kld_loss, dim=1))
            kld_loss /= 2
            kld_loss *= 1.0

            total_loss = ce_loss + kld_loss

            grad = torch.autograd.grad(total_loss, inputs_embeds,
                                       retain_graph=False, create_graph=False)[0]

            adv_embeds = inputs_embeds + 0.005 * grad.sign()


        # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(
            input_ids if not adv_training else None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=adv_embeds if adv_training else None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # [batch_size, seq_len, hidden_size]
        sequence_output = outputs[0]
        # [batch_size, seq_len, 2]
        logits = self.classifier(sequence_output)
        # [batch_size, seq_len, 1]
        start_logits, end_logits = logits.split(1, dim=-1)
        # [batch_size, seq_len]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,)
        if start_position is not None and end_position is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_position.size()) > 1:
                start_position = start_position.squeeze(-1)
            if len(end_position.size()) > 1:
                end_position = end_position.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_position.clamp_(0, ignored_index)
            end_position.clamp_(0, ignored_index)

            # ce loss
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_position)
            end_loss = loss_fct(end_logits, end_position)
            ce_loss = (start_loss + end_loss) / 2

            # kld loss
            start_kld_loss = kld_loss_func(log_softmax(start_logits), softmax(teacher_start_logits / temp))
            kld_loss = torch.mean(torch.sum(start_kld_loss, dim=1))
            end_kld_loss = kld_loss_func(log_softmax(end_logits), softmax(teacher_end_logits / temp))
            kld_loss += torch.mean(torch.sum(end_kld_loss, dim=1))
            kld_loss /= 2
            kld_loss *= 1.0

            total_loss = ce_loss + kld_loss

            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)