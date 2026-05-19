import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - depends on the local environment
    triton = None
    tl = None
    TRITON_IMPORT_ERROR = exc
else:
    TRITON_IMPORT_ERROR = None


def cuda_device() -> torch.device:
    """Return the active CUDA device or raise a clear project-level error."""
    if not torch.cuda.is_available():
        raise RuntimeError("The Triton implementation needs a CUDA GPU, but torch.cuda.is_available() is False.")
    return torch.device("cuda")


def require_triton() -> None:
    """Raise a helpful error if Triton could not be imported."""
    if TRITON_IMPORT_ERROR is not None:
        raise RuntimeError("Triton is not installed in this Python environment.") from TRITON_IMPORT_ERROR


if triton is not None:

    @triton.jit
    def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        GROUPED: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

        if GROUPED:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
        else:
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_mask = offs_k < K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b, input_precision="tf32")
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask)


    @triton.jit
    def bias_add_kernel(x_ptr, b_ptr, out_ptr, total, q: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        cols = offsets % q
        x = tl.load(x_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + cols, mask=mask)
        tl.store(out_ptr + offsets, x + b, mask=mask)


    @triton.jit
    def fused_softmax_crossentropy_kernel(
        logits_ptr,
        labels_ptr,
        probs_ptr,
        losses_ptr,
        n_cols: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        logits = tl.load(logits_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
        shifted = logits - tl.max(logits, axis=0)
        numer = tl.exp(shifted)
        probs = numer / tl.sum(numer, axis=0)
        tl.store(probs_ptr + row * n_cols + cols, probs, mask=mask)

        label = tl.load(labels_ptr + row)
        correct_prob = tl.sum(tl.where(cols == label, probs, 0.0), axis=0)
        loss = -tl.log(tl.maximum(correct_prob, 1.0e-20))
        tl.store(losses_ptr + row, loss)


    @triton.jit
    def softmax_only_kernel(logits_ptr, probs_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        logits = tl.load(logits_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
        shifted = logits - tl.max(logits, axis=0)
        numer = tl.exp(shifted)
        probs = numer / tl.sum(numer, axis=0)
        tl.store(probs_ptr + row * n_cols + cols, probs, mask=mask)


    @triton.jit
    def crossentropy_only_kernel(probs_ptr, labels_ptr, losses_ptr, n_cols: tl.constexpr):
        row = tl.program_id(0)
        label = tl.load(labels_ptr + row)
        prob = tl.load(probs_ptr + row * n_cols + label)
        tl.store(losses_ptr + row, -tl.log(tl.maximum(prob, 1.0e-20)))


    @triton.jit
    def softmax_ce_backward_kernel(
        probs_ptr,
        labels_ptr,
        grad_ptr,
        total,
        n_cols: tl.constexpr,
        inv_batch: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        cols = offsets % n_cols
        rows = offsets // n_cols
        labels = tl.load(labels_ptr + rows, mask=mask, other=-1)
        probs = tl.load(probs_ptr + offsets, mask=mask)
        one_hot = cols == labels
        tl.store(grad_ptr + offsets, (probs - one_hot) * inv_batch, mask=mask)


    @triton.jit
    def row_sum_kernel(x_ptr, out_ptr, n_rows: tl.constexpr, n_cols: tl.constexpr, BLOCK_M: tl.constexpr):
        col = tl.program_id(0)
        rows = tl.arange(0, BLOCK_M)
        mask = rows < n_rows
        vals = tl.load(x_ptr + rows * n_cols + col, mask=mask, other=0.0)
        tl.store(out_ptr + col, tl.sum(vals, axis=0))


    @triton.jit
    def sgd_update_kernel(param_ptr, grad_ptr, n_elements, lr: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        param = tl.load(param_ptr + offsets, mask=mask)
        grad = tl.load(grad_ptr + offsets, mask=mask)
        tl.store(param_ptr + offsets, param - lr * grad, mask=mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor, grouped: bool = True) -> torch.Tensor:
    """Compute C = A @ B with the FP32 Triton matmul kernel.

    Args:
        a: Contiguous CUDA tensor with shape (M, K) and dtype float32.
        b: Contiguous CUDA tensor with shape (K, N) and dtype float32.
        grouped: Whether to use grouped program ordering for better L2 reuse.
    """
    require_triton()
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        GROUPED=grouped,
        num_warps=4,
        num_stages=4,
    )
    return c


def triton_bias_add(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Add a class-bias vector to every row of a logits matrix."""
    require_triton()
    out = torch.empty_like(x)
    total = x.numel()
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
    bias_add_kernel[grid](x, b, out, total, x.shape[1], BLOCK_SIZE=1024)
    return out


def triton_fused_softmax_ce(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-example cross-entropy and probabilities in one row-wise kernel."""
    require_triton()
    n_rows, n_cols = logits.shape
    probs = torch.empty_like(logits)
    losses = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    block = triton.next_power_of_2(n_cols)
    fused_softmax_crossentropy_kernel[(n_rows,)](logits, labels, probs, losses, n_cols, BLOCK_SIZE=block, num_warps=1)
    return losses, probs


def triton_unfused_softmax_ce(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute softmax and cross-entropy with two separate Triton kernels."""
    require_triton()
    n_rows, n_cols = logits.shape
    probs = torch.empty_like(logits)
    losses = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    block = triton.next_power_of_2(n_cols)
    softmax_only_kernel[(n_rows,)](logits, probs, n_cols, BLOCK_SIZE=block, num_warps=1)
    crossentropy_only_kernel[(n_rows,)](probs, labels, losses, n_cols, num_warps=1)
    return losses, probs


def triton_softmax_ce_backward(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return d_logits = (softmax(logits) - one_hot(labels)) / batch_size."""
    require_triton()
    grad = torch.empty_like(probs)
    total = probs.numel()
    inv_batch = 1.0 / probs.shape[0]
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
    softmax_ce_backward_kernel[grid](
        probs,
        labels,
        grad,
        total,
        probs.shape[1],
        inv_batch,
        BLOCK_SIZE=1024,
    )
    return grad


def triton_row_sum(x: torch.Tensor) -> torch.Tensor:
    """Reduce a small batch matrix along rows to form the bias gradient."""
    require_triton()
    n_rows, n_cols = x.shape
    if n_rows > 1024:
        raise ValueError("row_sum_kernel is intentionally simple for this course project; use batch_size <= 1024.")
    out = torch.empty((n_cols,), device=x.device, dtype=torch.float32)
    row_sum_kernel[(n_cols,)](x, out, n_rows, n_cols, BLOCK_M=triton.next_power_of_2(n_rows), num_warps=8)
    return out


def triton_sgd_update(param: torch.Tensor, grad: torch.Tensor, lr: float) -> None:
    """Apply an in-place SGD update: param -= lr * grad."""
    require_triton()
    total = param.numel()
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
    sgd_update_kernel[grid](param, grad, total, lr, BLOCK_SIZE=1024)
