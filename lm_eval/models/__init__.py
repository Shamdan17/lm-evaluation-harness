# from . import anthropic_llms
# from . import huggingface
# from . import hf_no_softmax
# from . import textsynth
# from . import deepsparse
# from . import dummy
# from . import gguf

# MODEL_REGISTRY = {
#     "hf": huggingface.HFLM,
#     "hf-causal": huggingface.HFLM,
#     "hf-causal-experimental": huggingface.AutoCausalLM,
#     "hf-seq2seq": huggingface.AutoSeq2SeqLM,
#     "hf-causal-no-softmax": hf_no_softmax.GloballyNormalizedCausalLM,
#     "hf-seq2seq-no-softmax": hf_no_softmax.GloballyNormalizedSeq2SeqLM,
#     "anthropic": anthropic_llms.AnthropicLM,
#     "textsynth": textsynth.TextSynthLM,
#     "deepsparse": deepsparse.DeepSparseLM,
#     "dummy": dummy.DummyLM,
#     "gguf": gguf.GGUFLM,
# }
from . import (
    anthropic_llms,
    dummy,
    gguf,
    huggingface,
    mamba_lm,
    neuron_optimum,
    openai_completions,
    optimum_lm,
    textsynth,
    vllm_causallms,
    hf_no_softmax,
)


# TODO: implement __all__


try:
    # enable hf hub transfer if available
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass
