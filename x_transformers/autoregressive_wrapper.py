from math import ceil, log
from typing import Optional, Union, Tuple, Callable

import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from einops import rearrange, pack, unpack

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def identity(t, *args, **kwargs):
    return t

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else (t,) * length

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# for variable lengthed prefixes

def align_right(t, lens, pad_id = 0):
    batch, seq_len, device, dtype = *t.shape, t.device, t.dtype

    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len

    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()

    batch_arange = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device = device, dtype = torch.long)

    t = F.pad(t, (max_pad_len, 0), value = 0)
    offset = max_pad_len - pad_lens

    aligned = t[batch_arange, prompt_len_arange + offset[..., None]]
    return aligned

# nucleus

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, frac_num_tokens = 0.1, k = None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# top_a

def top_a(logits, min_p_pow = 2.0, min_p_ratio = 0.02):
    probs = F.softmax(logits, dim = -1)
    max_probs = torch.amax(probs, dim = -1, keepdim = True)
    limit = torch.pow(max_probs, min_p_pow) * min_p_ratio
    return torch.where(probs < limit, float('-inf'), logits)

# contrastive decoding function

def contrastive_decode_fn(
    expert_logits,
    amateur_logits,
    alpha = 0.1,
    beta = 0.5
):
    """
    Appendix A Algorithm 2
    https://arxiv.org/abs/2309.09117
    """

    cutoff = log(alpha) + expert_logits.amax(dim = -1, keepdim = True)
    diffs = (1 + beta) * expert_logits - beta * amateur_logits
    contrastive_decode_logits = diffs.masked_fill(expert_logits < cutoff, -torch.finfo(expert_logits.dtype).max)
    return contrastive_decode_logits

# beam search

def beam_search(
    model, 
    X, 
    predictions = 20,
    beam_width = 10,
):
    """
    Implements Beam Search to extend the sequences given in X. The method can compute 
    several outputs in parallel with the first dimension of X.

    Parameters
    ----------    
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.

    predictions: int
        The number of tokens to append to X.

    beam_width: int
        The number of candidates to keep in the search.

    Returns
    -------
    X: LongTensor of shape (examples, length + predictions)
        The sequences extended with the decoding process.

    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
        probability of the next token at every step.
    """
    with torch.no_grad():
        # The next command can be a memory bottleneck, but can be controlled with the batch 
        # size of the predict method.
        next_probabilities = model.forward(X)[:, -1, :]
        B, vocabulary_size = next_probabilities.shape
        probabilities, idx = next_probabilities.squeeze().log_softmax(-1)\
            .topk(k = beam_width, axis = -1)
        X = X.repeat((beam_width, 1, 1)).transpose(0, 1)\
            .flatten(end_dim = -2)
        next_chars = idx.reshape(-1, 1)
        X = torch.cat((X, next_chars), axis = -1)
        # This has to be minus one because we already produced a round
        # of predictions before the for loop.
        predictions_iterator = range(predictions - 1)
        for i in predictions_iterator:
            dataset = TensorDataset(X)
            loader = DataLoader(dataset, batch_size = B)
            next_probabilities = []
            iterator = iter(loader)
            for (x,) in iterator:
                next_probabilities.append(
                    model.forward(x)[:, -1, :].log_softmax(-1)
                )
            next_probabilities = torch.cat(next_probabilities, axis = 0)
            next_probabilities = next_probabilities.reshape(
                (-1, beam_width, next_probabilities.shape[-1])
            )
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities = probabilities.flatten(start_dim = 1)
            probabilities, idx = probabilities.topk(
                k = beam_width, 
                axis = -1
            )
            next_chars = torch.remainder(idx, vocabulary_size).flatten()\
                .unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            best_candidates += torch.arange(
                X.shape[0] // beam_width, 
                device = X.device
            ).unsqueeze(-1) * beam_width
            X = X[best_candidates].flatten(end_dim = -2)
            X = torch.cat((X, next_chars), axis = 1)
        return X.reshape(-1, beam_width, X.shape[-1]), probabilities

# autoregressive wrapper class

class AutoregressiveWrapper(Module):
    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0,
        bos_value = 1,
        eos_value = 2,
        mask_prob = 0.,
        add_attn_z_loss = False,
    ):
        super().__init__()
        self.pad_value = pad_value
        self.bos_token = bos_value
        self.eos_token = eos_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big improvements https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.
        self.mask_prob = mask_prob

        # whether to add router z-loss
        self.add_attn_z_loss = add_attn_z_loss

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompts,
        seq_len,
        eos_token = None,
        temperature = 1.,
        prompt_lens: Optional[Tensor] = None,
        filter_logits_fn: Callable = top_k,
        restrict_to_max_seq_len = True,
        amateur_model: Optional[Union[Module, Tuple[Module]]] = None,
        filter_kwargs: dict = dict(),
        contrastive_decode_kwargs: Union[dict, Tuple[dict]] = dict(
            beta = 0.5,
            alpha = 0.1
        ),
        cache_kv = True,
        **kwargs
    ):
        max_seq_len, greedy, device = self.max_seq_len, temperature == 0., prompts.device

        prompts, ps = pack([prompts], '* n')

        b, t = prompts.shape

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id = self.pad_value)
            seq_start_pos = t - prompt_lens

        # output from which sampled tokens appended to

        out = prompts

        # kv caches

        cache = None

        # if doing contrastive decoding, turn off filter automatically

        if exists(amateur_model):
            amateur_model = cast_tuple(amateur_model)
            contrastive_decode_kwargs = cast_tuple(contrastive_decode_kwargs)

            assert len(amateur_model) == len(contrastive_decode_kwargs)

            amateur_caches = [None] * len(amateur_model)
            filter_logits_fn = identity

            for i, module in enumerate(amateur_model):
                if isinstance(module, AutoregressiveWrapper):
                    amateur_model[i] = module.net

                module.eval()

        # sampling up to seq_len

        for _ in range(seq_len):

            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len

                assert not (cache_kv and max_len_exceeded and not self.net.can_cache_kv_outside_max_seq_len), 'the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embeeding. you can switch to rotary embeddings to resolve this issue'

                x = out[:, -max_seq_len:]

                if exists(cache):
                    for inter in cache.attn_intermediates:
                        inter.cached_kv = [t[..., -(max_seq_len - 1):, :] for t in inter.cached_kv]

            logits, new_cache = self.net(
                x,
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos,
                **kwargs
            )

            if cache_kv and self.net.can_cache_kv:
                cache = new_cache

            logits = logits[:, -1]

            # handle contrastive decoding, Li et al.
            # https://arxiv.org/abs/2210.15097

            if exists(amateur_model):
                for i, (amateur, amateur_cache, amateur_contrastive_decode_kwargs) in enumerate(zip(amateur_model, amateur_caches, contrastive_decode_kwargs)):
                    amateur_logits, next_amateur_cache = amateur(
                        x,
                        return_intermediates = True,
                        cache = amateur_cache,
                        seq_start_pos = seq_start_pos,
                        **kwargs
                    )

                    amateur_logits = amateur_logits[:, -1]

                    assert amateur_logits.shape == logits.shape, 'logits dimension are not the same between amateur and expert model'
                    logits = contrastive_decode_fn(logits, amateur_logits, **amateur_contrastive_decode_kwargs)

                    if cache_kv and amateur.can_cache_kv:
                        amateur_caches[i] = next_amateur_cache

            # filter by top_k, top_p (nucleus), top_a, or custom

            if greedy:
                sample = logits.argmax(dim = -1, keepdim = True)
            else:
                filtered_logits = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            # concat sample

            out = torch.cat((out, sample), dim=-1)

            if (out == self.eos_token).all() or (out == self.pad_value).all():
                break

        if exists(eos_token):
            # mask out everything after the eos tokens
            is_eos_tokens = (out == eos_token)
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        out = out[:, t:]

        out, = unpack(out, ps, '* n')

        return out
    
    @torch.no_grad()
    @eval_decorator
    def beam_generate(
        self, 
        enc, 
        enc_mask = None,
        beam_width=3,
    ):
        B, device = enc.shape[0], enc.device
        # self.net(x, context = enc, context_mask = enc_mask)
        
        # Initialize the scores for each beam with zeros. Shape: [beam_size]
        beam_scores = torch.zeros(beam_width).to(device)
        
        # Duplicate the initial sequence for each beam. Shape: [beam_width, seq_length]
        beam_sequences = torch.tensor(self.bos_token).unsqueeze(0).unsqueeze(0).to(device)
        
        # Create a boolean mask to keep track of active beams. Shape: [beam_width]
        active_beams = torch.ones(beam_width, dtype=torch.bool)
    
        
        for step in range(self.max_seq_len):
            # Generate model outputs for the current sequences. 
            # The model is expected to handle batched input, hence the shape of beam_sequences is [beam_width, current_seq_length].
            if enc.shape[0] != beam_sequences.shape[0]:
                enc = enc.repeat(beam_width, 1, 1)
                enc_mask = enc_mask.repeat(beam_width, 1)
            outputs = self.net(beam_sequences, context = enc, context_mask = enc_mask)
            
            # Extract the last logits from the output to get the probabilities for the next token. Shape: [beam_width, vocab_size]
            logits = outputs[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Calculate the score for each beam and token by flattening the probabilities and selecting the top ones.
            # The flattened shape is [beam_width * vocab_size], from which we select the top beam_width scores.
            top_scores, top_indices = torch.topk(probs.flatten(), k=beam_width, sorted=False)
        
            # Map flat indices back to beam and token indices.
            # beam_indices is the index in the beam, shape: [beam_width]
            # token_indices is the index of the token in the vocabulary, shape: [beam_width]
            beam_indices = top_indices // probs.shape[-1]
            token_indices = top_indices % probs.shape[-1]
            
            # Update the sequences with the new tokens at the end. Shape after update: [beam_width, current_seq_length + 1]
            # This concatenates the best token for each beam to the end of the sequences.
            beam_sequences = torch.cat([
                beam_sequences[beam_indices],
                token_indices.unsqueeze(-1)
            ], dim=-1)

            # Update the beam scores with the top scores. Shape: [beam_width]
            beam_scores = top_scores
            
            # Check for the end-of-sequence tokens and update the active beams.
            # If a beam produces an EOS token, it is marked as inactive.
            active_beams = ~(token_indices == self.eos_token)
            
            # If all beams are inactive, exit the loop.
            if not active_beams.any():
                break
            
        # sort the beams by their scores
        best_beams = beam_scores.argsort(descending=True)
        best_scores = beam_scores[best_beams[:beam_width]]
        best_sequences = beam_sequences[best_beams[:beam_width]]

        # mask out everything after the eos tokens
        is_eos_tokens = (best_sequences == self.eos_token)
        shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
        mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
        best_sequences = best_sequences.masked_fill(mask, self.pad_value)
        
        return best_sequences, best_scores

    
    def forward(self, x, return_outputs = False, **kwargs):
        seq, ignore_index, add_attn_z_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss

        inp, target = x[:, :-1], x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        if self.mask_prob > 0.:
            rand = torch.randn(inp.shape, device = x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max # first token should not be masked out
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim = -1).indices
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.).bool()
            kwargs.update(self_attn_kv_mask = mask)

        logits, cache = self.net(
            inp,
            return_intermediates = True,
            return_attn_z_loss = add_attn_z_loss,
            **kwargs
        )

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index
        )

        if add_attn_z_loss:
            loss = loss + cache.attn_z_loss

        if not return_outputs:
            return loss

        return loss, (logits, cache)
