import torch


def squeeze_input_for_thd(input_ids, position_ids, padding_mask, attn_kwargs, seqlens_padding_value=-1000):
    input_ids = input_ids.squeeze(0)
    position_ids = position_ids.squeeze(0)
    padding_mask = padding_mask.squeeze(0)
    for key, value in attn_kwargs.items():
        if isinstance(value, torch.Tensor):
            attn_kwargs[key] = value.squeeze(0)
        if key in ["cu_seqlens", "cu_seqlens_padded"]:
            attn_kwargs[key] = value[value != seqlens_padding_value].contiguous()
        if key == "max_seqlen" and isinstance(value, torch.Tensor):
            attn_kwargs[key] = value.item()

    return input_ids, position_ids, padding_mask, attn_kwargs
