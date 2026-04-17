import inspect

from transformers import AutoTokenizer


def _supported_kwargs(callable_obj, kwargs):
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


class VLLMWrapper(object):
    def __init__(self, args, model_dir, lora_adapter_path=None, memory_for_model_activations_in_gb=2, dtype="float16"):
        super(VLLMWrapper, self).__init__()
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "vLLM is required for the default inference backend. "
                "Install requirements.txt or run with --inference_backend transformers."
            ) from exc

        if lora_adapter_path is not None:
            raise NotImplementedError("LoRA adapter loading is not wired for the vLLM inference backend.")

        self.name = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            token=getattr(args, "token", None),
            cache_dir=getattr(args, "model_dir", None),
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        llm_kwargs = {
            "model": model_dir,
            "tokenizer": model_dir,
            "download_dir": getattr(args, "model_dir", None),
            "dtype": dtype,
            "trust_remote_code": True,
            "tensor_parallel_size": getattr(args, "tensor_parallel_size", 1),
            "gpu_memory_utilization": getattr(args, "vllm_gpu_memory_utilization", 0.9),
            "seed": getattr(args, "seed", 42),
            "hf_token": getattr(args, "token", None),
        }
        max_model_len = getattr(args, "max_model_len", None)
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**_supported_kwargs(LLM, llm_kwargs))
        self.huggingface_model = None

    def generate(self, args, query, max_new_tokens=64, return_logits=True, verbose=False):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=None,
        )
        outputs = self.llm.generate(query, sampling_params)
        return [output.outputs[0].text for output in outputs]
