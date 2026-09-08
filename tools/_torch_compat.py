"""torch<2.4 compatibility shims (macOS x86_64 caps at torch 2.2.2)."""

import torch  # pyright: ignore[reportMissingImports]

def _install_rmsnorm() -> None:
    """Shim nn.RMSNorm for torch<2.4 (macOS x86_64 caps at 2.2.2).

    Exact replica of torch>=2.4 semantics: eps=None uses
    torch.finfo(x.dtype).eps (f32 ≈ 1.19e-7), weight initialized to ones.
    Must be installed before importing the official model code.
    """

    if hasattr(torch.nn, "RMSNorm"):  # pragma: no cover
        return

    class RMSNorm(torch.nn.Module):
        def __init__(
            self,
            normalized_shape: int | tuple[int, ...],
            eps: float | None = None,
            elementwise_affine: bool = True,
        ) -> None:
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = torch.nn.Parameter(
                    torch.ones(self.normalized_shape)
                )
            else:
                self.register_parameter("weight", None)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            eps = self.eps
            if eps is None:
                eps = torch.finfo(x.dtype).eps
            ndim = len(self.normalized_shape)
            axes = tuple(range(-ndim, 0))
            rms = x.pow(2).mean(dim=axes, keepdim=True)
            out = x * torch.rsqrt(rms + eps)
            if self.weight is not None:
                out = out * self.weight
            return out

    torch.nn.RMSNorm = RMSNorm  # type: ignore[attr-defined]


_install_rmsnorm()
