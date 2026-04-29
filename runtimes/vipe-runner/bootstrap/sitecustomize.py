import torch


torch.backends.cudnn.enabled = False

_orig_quantile = torch.quantile


def _quantile_with_cpu_fallback(input_tensor, q, *args, **kwargs):
    try:
        return _orig_quantile(input_tensor, q, *args, **kwargs)
    except RuntimeError as exc:
        if "quantile() input tensor is too large" not in str(exc):
            raise
        cpu_input = input_tensor.detach().cpu()
        cpu_q = q.detach().cpu() if torch.is_tensor(q) else q
        result = _orig_quantile(cpu_input, cpu_q, *args, **kwargs)
        if torch.is_tensor(result):
            result = result.to(device=input_tensor.device)
            if input_tensor.is_floating_point():
                result = result.to(dtype=input_tensor.dtype)
        return result


def _tensor_quantile_with_cpu_fallback(self, q, *args, **kwargs):
    return torch.quantile(self, q, *args, **kwargs)


torch.quantile = _quantile_with_cpu_fallback
torch.Tensor.quantile = _tensor_quantile_with_cpu_fallback
