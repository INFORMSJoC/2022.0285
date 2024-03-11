import os
import torch
import numpy as np
import random
import opencc
converter = opencc.OpenCC('t2s')
import torch.nn as nn

# Set seed for reproducibility
def seed_everything(seed=0):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:2"
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False



# Encoding function for cross attention based models.
def encode(tokenizer, subreddit_dict, input_dict, max_length, clause_max_length):
    def encode_words(tokenizer, words):
        tokens = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
        return tokens
    sarcasm = input_dict["sarcasm"][:-5] + input_dict["sarcasm"][-5:].replace("/s", "")
    sarcasm = sarcasm.strip()
    sarcasm_words = sarcasm.split()
    subreddit = input_dict["subreddit"]
    topic_words = subreddit_dict[subreddit]
    clauses = input_dict["clauses"]
    labels = input_dict["clause_labels"]
    clause_idx = []
    clause_labels = []
    token_type_id = 0
    tokens = []
    token_type_ids = []
    attention_mask = []
    input_mask = []
    position_ids = []
    real_len = 0
    max_pre_length = max_length - clause_max_length
    max_clause_id = 0
    for id_,(clause, label) in enumerate(zip(clauses, labels)):
        clause_words = clause.split()
        clause_tokens = encode_words(tokenizer, clause_words)
        # Note: The first token of each clause is [PAIR], which means the representation is learned for pairwise task
        # The [CLS] token is servered as the sentence level representation.
        clause_tokens = [tokenizer.pair_token] + clause_tokens[:clause_max_length - 2] + [
            tokenizer.sep_token]
        # Exceed max_len
        if real_len + len(clause_tokens) > max_pre_length:
            # Can insert some words
            if real_len < max_pre_length - 2:
                clause_tokens = clause_tokens[:max_pre_length - real_len - 1] + [tokenizer.sep_token]
                clause_idx.append(id_)
                max_clause_id = id_
                clause_labels.append(label)
                tokens.extend(clause_tokens + [tokenizer.pad_token] * (clause_max_length - len(clause_tokens)))
                token_type_ids.extend([token_type_id] * clause_max_length)
                attention_mask.extend([1] * len(clause_tokens) + [0] * (clause_max_length - len(clause_tokens)))
                input_mask.extend([0] + [1] * (len(clause_tokens) - 1) + [0] * (clause_max_length - len(clause_tokens)))
                position_ids.extend(list(range(real_len, real_len + len(clause_tokens))) + [max_length] * (clause_max_length - len(clause_tokens)))
                real_len += len(clause_tokens)
                token_type_id = 1 - token_type_id
            break
        else:
            clause_idx.append(id_)
            max_clause_id = id_
            clause_labels.append(label)
            tokens.extend(clause_tokens + [tokenizer.pad_token] * (clause_max_length - len(clause_tokens)))
            token_type_ids.extend([token_type_id] * clause_max_length)
            attention_mask.extend([1] * len(clause_tokens) + [0] * (clause_max_length - len(clause_tokens)))
            input_mask.extend([0] + [1] * (len(clause_tokens) - 1) + [0] * (clause_max_length - len(clause_tokens)))
            position_ids.extend(
                list(range(real_len, real_len + len(clause_tokens))) + [max_length] * (clause_max_length - len(clause_tokens)))
            real_len += len(clause_tokens)
            token_type_id = 1 - token_type_id

    assert real_len <= max_pre_length
    assert len(tokens) == len(attention_mask) == len(token_type_ids) == len(position_ids) == len(input_mask)
    assert len(tokens) % clause_max_length == 0
    assert len(clause_idx) == len(clause_labels)

    sarcasm_idx = [max_clause_id + 1]
    sarcasm_tokens = encode_words(tokenizer, sarcasm_words)
    sarcasm_tokens = [tokenizer.pair_token] + sarcasm_tokens[:clause_max_length - 2] + [tokenizer.sep_token]
    position_ids.extend(
        list(range(real_len, real_len + len(sarcasm_tokens))) + [max_length] * (clause_max_length - len(sarcasm_tokens)))
    real_len += len(sarcasm_tokens)
    tokens.extend(sarcasm_tokens + [tokenizer.pad_token] * (clause_max_length - len(sarcasm_tokens)))
    attention_mask.extend([1] * len(sarcasm_tokens) + [0] * (clause_max_length - len(sarcasm_tokens)))
    input_mask.extend([0] + [1] * (len(sarcasm_tokens) - 1) + [0] * (clause_max_length - len(sarcasm_tokens)))
    token_type_ids.extend([token_type_id] * clause_max_length)

    assert real_len <= max_length
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    input_mask = torch.tensor(input_mask)
    token_type_ids = torch.tensor(token_type_ids)
    position_ids = torch.tensor(position_ids)
    clause_idx = torch.tensor(clause_idx)
    clause_labels = torch.tensor(clause_labels)
    sarcasm_idx = torch.tensor(sarcasm_idx)

    # For symmetry, we still use [PAIR] token to carry sentence level information
    subreddit_tokens = [tokenizer.pair_token] + tokenizer.tokenize(topic_words)[:clause_max_length- 2] + [tokenizer.sep_token]
    token_type_id = 0
    subreddit_token_type_ids = [token_type_id] * len(subreddit_tokens)
    subreddit_attention_mask = [1] * len(subreddit_tokens)
    subreddit_position_ids = list(range(len(subreddit_attention_mask)))

    # Pad
    subreddit_tokens.extend([tokenizer.pad_token] * (clause_max_length - len(subreddit_tokens)))
    subreddit_attention_mask.extend([0] * (clause_max_length - len(subreddit_attention_mask)))
    subreddit_token_type_ids.extend([0] * (clause_max_length - len(subreddit_token_type_ids)))
    subreddit_position_ids.extend([max_length] * (clause_max_length - len(subreddit_position_ids)))
    subreddit2ids = {'android': 0, 'apple': 1, 'technology': 2, 'dota2': 3, 'playstation': 4, 'movies': 5, 'nba': 6, 'steam': 7}
    # print(subreddit)
    subreddit_idx = [subreddit2ids[subreddit]]
    subreddit_idx = torch.tensor(subreddit_idx)
    subreddit_input_ids = tokenizer.convert_tokens_to_ids(subreddit_tokens)

    subreddit_input_ids = torch.tensor(subreddit_input_ids)
    subreddit_attention_mask = torch.tensor(subreddit_attention_mask)
    subreddit_token_type_ids = torch.tensor(subreddit_token_type_ids)
    subreddit_position_ids = torch.tensor(subreddit_position_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "input_mask": input_mask,
            "token_type_ids": token_type_ids, "position_ids": position_ids,
            "subreddit_input_ids": subreddit_input_ids, "subreddit_attention_mask": subreddit_attention_mask,
            "subreddit_token_type_ids": subreddit_token_type_ids, "subreddit_position_ids": subreddit_position_ids,
            "clause_idx": clause_idx, "sarcasm_idx": sarcasm_idx, "subreddit_idx": subreddit_idx,
            "clause_labels": clause_labels}

# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)


# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


# Code inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.
    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.
    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add

def sent_metrics(preds, truths):
    corrects = preds.intersection(truths)
    if len(corrects) == 0:
        return 0.0, 0.0, 0.0
    precision = len(corrects) / len(preds)
    recall = len(corrects) / len(truths)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

