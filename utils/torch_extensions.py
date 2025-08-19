from typing import List
import torch


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


@staticmethod
def _build_block_diag_mask(hist_len: int,
                           render_lens: List[int],
                           dtype=torch.float32,
                           device: torch.device = torch.device("cpu")
                           ) -> torch.Tensor:
    """
    生成形状 (1,1,L,L) 的 block-diag causal mask：
      · 元素 = 0      → 允许注意力
      · 元素 = -1e4   → 禁止注意力
    其中
      hist_len      = KB + history 这部分 token 总长度
      render_lens   = [len(render₀), len(render₁), …]
    """
    L = hist_len + sum(render_lens)
    m = torch.zeros((1, 1, L, L), dtype=dtype, device=device)

    # ① causal
    i, j = torch.triu_indices(L, L, 1)
    m[0, 0, i, j] = -1e4

    # ② 阻断不同 render 之间互相注意
    ptr, spans = hist_len, []
    for ln in render_lens:
        spans.append((ptr, ptr + ln))
        ptr += ln

    for a, (s_a, e_a) in enumerate(spans):
        for b, (s_b, e_b) in enumerate(spans):
            if a != b:
                m[0, 0, s_a:e_a, s_b:e_b] = -1e4
    return m
