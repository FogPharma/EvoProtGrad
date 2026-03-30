from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from pplm.models.hf_models import CrossAttnRankingModel, PplmHitRankingModel
from transformers import AutoTokenizer, EsmForMaskedLM, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding

import evo_prot_grad.common.embeddings as embeddings
from evo_prot_grad.experts.base_experts import ProteinLMExpert
from evo_prot_grad.experts.esm_expert import EsmExpert


class PplmExpert(EsmExpert):
    """Expert baseclass for HuggingFace protein language models from the ESM family.
    Implements abstract methods `_get_last_one_hots` and `tokenize`.
    Swaps out the `EsmForMaskedLM.esm.embeddings.word_embeddings` layer
    for a `evo_prot_grad.common.embeddings.OneHotEmbedding` layer.
    """

    def __init__(
        self,
        temperature: float,
        scoring_strategy: str,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        device: str = "cpu",
        target_embed_idx: Optional[int] = None,
    ):
        """
        Args:
            temperature (float): Temperature for sampling from the expert.
            scoring_strategy (str): Approach for scoring variants that the expert will use.
            model (nn.Module): The model to use for the expert. Defaults to EsmForMaskedLM from facebook/esm2_t6_8M_UR50D.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use for the expert. Defaults to AutoTokenizer from facebook/esm2_t6_8M_UR50D.
            device (str): The device to use for the expert. Defaults to 'cpu'.
        Raises:
            ValueError: If either `model` or `tokenizer` is not specified.
        """
        if model is None or tokenizer is None:
            raise ValueError(
                "PplmExpert requires both `model` and `tokenizer` to be specified."
            )
        super().__init__(
            temperature=temperature,
            scoring_strategy=scoring_strategy,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        self.pplm_model_type_ = (
            "cross_attn" if isinstance(model, CrossAttnRankingModel) else "hit"
        )
        if target_embed_idx is None and self.pplm_model_type_ == "cross_attn":
            raise ValueError("CrossAttnRankingModel requires integer `target_embed_idx")
        self.target_idx = target_embed_idx

    def get_model_output(
        self, inputs: List[str], target_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the one-hot sequences and logits for each amino acid in the
           input sequence.

        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            x_oh: (torch.Tensor) of shape [parallel_chains, seq_len, vocab_size]
            logits: (torch.Tensor) of shape [parallel_chains, seq_len, vocab_size]
        """
        encoded_inputs = self.tokenize(inputs)
        if self.pplm_model_type_ == "cross_attn":
            target_tensor = torch.LongTensor([self.target_idx] * len(inputs))
            logits = self.model(**encoded_inputs, target=target_tensor)
        else:
            logits = self.model(**encoded_inputs).logits
        # pplm ranking models return a B x 3 tensor, with the following dimensions:
        #  0: p(not hit)
        #  1: p(hit)
        #  2: predicted score
        # p_hit = logits[:, 1]
        oh = self._get_last_one_hots()
        return oh, logits.view(-1)

    def __call__(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the one-hot sequences and expert score.
        Assumes the pLM predicts a logit score for each amino acid.

        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            oh (torch.Tensor): of shape [parallel_chains, seq_len, vocab_size]
            expert_score (torch.Tensor): of shape [parallel_chains]
        """
        oh, logits = self.get_model_output(inputs)
        score = self.variant_scoring(oh, logits, self._wt_oh)
        return oh, score


def build(**kwargs):
    """Builds a Esm2Expert."""
    return PplmExpert(**kwargs)
