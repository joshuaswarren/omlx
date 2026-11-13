# SPDX-License-Identifier: Apache-2.0
"""Untied output head for mlx-vlm Gemma 4.

mlx-vlm's Gemma 4 always computes logits as ``embed_tokens.as_linear`` and has
no ``lm_head``. A checkpoint with ``tie_word_embeddings: false`` stores a
separate top-level ``lm_head`` (``weight``, plus ``scales`` and ``biases`` when
quantized), so loading fails with "parameters not in model: lm_head.*" (#5170).

The patch gives such a model a top-level ``lm_head`` linear, the same layout
mlx-lm's ``gemma4_text`` uses, so the loader's quantization predicate and the
weight keys match. The language model reads it for logits. Tied checkpoints are
untouched.
"""

import mlx.nn as nn

_PATCHED = False


def _untied(config) -> bool:
    return config.tie_word_embeddings is False or (
        getattr(config.text_config, "tie_word_embeddings", True) is False
    )


def apply_gemma4_untied_lm_head_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return False
    try:
        from mlx_vlm.models.gemma4 import gemma4, language
    except ImportError:
        return False

    model_cls = gemma4.Model
    language_cls = language.LanguageModel
    original_init = model_cls.__init__
    original_logits = language_cls.logits_from_hidden

    def init_with_lm_head(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        if not _untied(config):
            return
        text = config.text_config
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        # A function attribute is not a Module parameter, so the head is not
        # registered twice.
        self.language_model._lm_head_of = lambda: self.lm_head

    def logits_from_hidden(self, hidden):
        head_of = getattr(self, "_lm_head_of", None)
        if head_of is None:
            return original_logits(self, hidden)
        logits = head_of()(hidden)
        if self.final_logit_softcapping is not None:
            logits = language.logit_softcap(self.final_logit_softcapping, logits)
        return logits

    model_cls.__init__ = init_with_lm_head
    language_cls.logits_from_hidden = logits_from_hidden
    _PATCHED = True
    return True
