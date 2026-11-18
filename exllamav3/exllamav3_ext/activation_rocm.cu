#if defined(USE_ROCM)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <cstdint>

#include "activation.cuh"
#include "util.h"
#include "util.cuh"

#define NUM_THREADS_P 1024

// x * sigmoid(y @ w) + z -> z
__global__ __launch_bounds__(NUM_THREADS_P)
void add_sigmoid_proj_kernel_f
(
    const float* __restrict__ px,
    const half* __restrict__ py,
    float* __restrict__ pz,
    const half* __restrict__ pw,
    const size_t dim
)
{
    const size_t b = blockIdx.x;
    const int t = threadIdx.x;
    const float* pxb = px + dim * b;
    const half* pyb = py + dim * b;
    float* pzb = pz + dim * b;

    float yw = 0.0f;
    for (size_t idx = t; idx < dim; idx += NUM_THREADS_P)
        yw += __half2float(pw[idx]) * __half2float(pyb[idx]);

    __shared__ float sums[NUM_THREADS_P];
    sums[t] = yw;
    __syncthreads();
    for (int stride = NUM_THREADS_P / 2; stride > 0; stride >>= 1)
    {
        if (t < stride) sums[t] += sums[t + stride];
        __syncthreads();
    }
    const float syw = 1.0f / (1.0f + __expf(-sums[0]));
    if (syw < 1e-8f) return;

    for (size_t idx = t; idx < dim; idx += NUM_THREADS_P)
        pzb[idx] += pxb[idx] * syw;
}

void add_sigmoid_gate_proj
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    const at::Tensor& w
)
{
    constexpr const char* op = "add_sigmoid_gate_proj";
    TORCH_CHECK(x.is_cuda(), op, ": x must be a CUDA tensor");
    const at::Device device = x.device();
    const auto check_device = [&](const at::Tensor& tensor)
    {
        TORCH_CHECK(tensor.is_cuda(), op, ": all tensors must be CUDA tensors");
        TORCH_CHECK(tensor.device() == device, op, ": all tensors must be on the same device");
    };
    check_device(y);
    check_device(z);
    check_device(w);

    TORCH_CHECK_DTYPE(x, kFloat);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK_DTYPE(z, kFloat);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK(x.dim() >= 1, op, ": x must have shape [..., D]");
    TORCH_CHECK(x.sizes() == y.sizes() && x.sizes() == z.sizes(),
                op, ": x, y and z must have identical shapes [..., D]");
    TORCH_CHECK(w.dim() == 2, op, ": w must have shape [D, 1]");

    const int64_t dim = x.size(-1);
    TORCH_CHECK(dim > 0, op, ": D must be greater than zero");
    TORCH_CHECK(w.size(0) == dim && w.size(1) == 1, op, ": w must have shape [D, 1]");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && z.is_contiguous() && w.is_contiguous(),
                op, ": all tensors must be contiguous");

    hipDeviceProp_t prop;
    cuda_check(hipGetDeviceProperties(&prop, device.index()));
    TORCH_CHECK(prop.maxThreadsPerBlock >= NUM_THREADS_P,
                op, ": device does not support the required block size");

    const size_t rows = static_cast<size_t>(x.numel() / dim);
    if (rows == 0) return;
    TORCH_CHECK(rows <= static_cast<size_t>(prop.maxGridSize[0]), op, ": too many rows");

    const c10::cuda::OptionalCUDAGuard device_guard(device);
    hipStream_t stream = c10::hip::getCurrentHIPStream(device.index()).stream();
    add_sigmoid_proj_kernel_f<<<rows, NUM_THREADS_P, 0, stream>>>
    (
        static_cast<const float*>(x.data_ptr()),
        static_cast<const half*>(y.data_ptr()),
        static_cast<float*>(z.data_ptr()),
        static_cast<const half*>(w.data_ptr()),
        static_cast<size_t>(dim)
    );
    cuda_check(hipPeekAtLastError());
}

// ---------------------------------------------------------------------------
// Elementwise output gates used by libtorch/attention.cu (compiled on ROCm).
// The CUDA-only activation.cu is excluded here; these mirror mul_sigmoid__gr /
// mul_sigmoid_broadcast__gr / mul_softplus_broadcast__gr and their kernels
// from activation.cu / activation_kernels.cuh.

#define NUM_THREADS 256

static __device__ __forceinline__ half2 _sigmoid_h2(half2 x)
{
    half2 one = __float2half2_rn(1.0f);
    half2 neg_x = __hneg2(x);
    half2 e = h2exp(neg_x);
    half2 sum = __hadd2(one, e);
    return h2rcp(sum);
}

// Numerically stable softplus, computed in fp32 (matches activation_kernels.cuh)
static __device__ __forceinline__ float _softplus_f32(float x)
{
    return fmaxf(x, 0.0f) + log1pf(__expf(-fabsf(x)));
}

__global__ __launch_bounds__(NUM_THREADS)
void mul_sigmoid_kernel_r
(
    half* __restrict__ x,
    const half* __restrict__ y,
    const size_t numel
)
{
    size_t idx = (blockIdx.x * NUM_THREADS + threadIdx.x);
    if (idx >= numel / 2) return;

    half2 x2 = ((half2*) x)[idx];
    half2 y2 = ((const half2*) y)[idx];
    x2 = __hmul2(x2, _sigmoid_h2(y2));
    ((half2*) x)[idx] = x2;
}

__global__ __launch_bounds__(NUM_THREADS)
void mul_sigmoid_broadcast_kernel_r
(
    half* __restrict__ x,
    const half* __restrict__ y,
    const size_t numel,
    const size_t dim
)
{
    size_t idx = (blockIdx.x * NUM_THREADS + threadIdx.x);
    if (idx >= numel / 2) return;

    size_t x_idx = idx * 2;
    size_t y_idx = x_idx / dim;
    half2 x2 = ((half2*) x)[idx];
    half2 y2 = __half2half2(y[y_idx]);
    x2 = __hmul2(x2, _sigmoid_h2(y2));
    ((half2*) x)[idx] = x2;
}

__global__ __launch_bounds__(NUM_THREADS)
void mul_softplus_broadcast_kernel_r
(
    half* __restrict__ x,
    const half* __restrict__ y,
    const size_t numel,
    const size_t dim
)
{
    size_t idx = (blockIdx.x * NUM_THREADS + threadIdx.x);
    if (idx >= numel / 2) return;

    size_t x_idx = idx * 2;
    size_t y_idx = x_idx / dim;
    half2 x2 = ((half2*) x)[idx];
    float g = _softplus_f32(__half2float(y[y_idx]));
    float2 xf = __half22float2(x2);
    xf.x *= g;
    xf.y *= g;
    ((half2*) x)[idx] = __float22half2_rn(xf);
}

void mul_sigmoid__gr
(
    at::Tensor& x,
    const at::Tensor& y,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    hipStream_t stream = graph ? graph->capture_stream : c10::hip::getCurrentHIPStream(x.device().index()).stream();

    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK_SHAPES_FULL(x, y);
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.numel() % 2 == 0, "x.numel() must be even");

    size_t numel = x.numel();
    size_t blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS);
    mul_sigmoid_kernel_r<<<blocks, NUM_THREADS, 0, stream>>>
    (
        (half*) x.data_ptr(),
        (const half*) y.data_ptr(),
        numel
    );

    cuda_check(cudaPeekAtLastError());
}

// x *= sigmoid(y), where x is [B, S, H, D] and y is [B, S, H]

void mul_sigmoid_broadcast__gr
(
    at::Tensor& x,
    const at::Tensor& y,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    hipStream_t stream = graph ? graph->capture_stream : c10::hip::getCurrentHIPStream(x.device().index()).stream();

    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK(x.dim() == 4, "x must be [B, S, H, D]");
    TORCH_CHECK(y.dim() == 3, "y must be [B, S, H]");
    TORCH_CHECK(x.size(0) == y.size(0), "x and y have incompatible shapes");
    TORCH_CHECK(x.size(1) == y.size(1), "x and y have incompatible shapes");
    TORCH_CHECK(x.size(2) == y.size(2), "x and y have incompatible shapes");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.size(3) % 2 == 0, "x.size(3) must be even");

    size_t numel = x.numel();
    size_t dim = x.size(3);
    size_t blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS);
    mul_sigmoid_broadcast_kernel_r<<<blocks, NUM_THREADS, 0, stream>>>
    (
        (half*) x.data_ptr(),
        (const half*) y.data_ptr(),
        numel,
        dim
    );

    cuda_check(cudaPeekAtLastError());
}

// x *= softplus(y), where x is [B, S, H, D] and y is [B, S, H]

void mul_softplus_broadcast__gr
(
    at::Tensor& x,
    const at::Tensor& y,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    hipStream_t stream = graph ? graph->capture_stream : c10::hip::getCurrentHIPStream(x.device().index()).stream();

    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK(x.dim() == 4, "x must be [B, S, H, D]");
    TORCH_CHECK(y.dim() == 3, "y must be [B, S, H]");
    TORCH_CHECK(x.size(0) == y.size(0), "x and y have incompatible shapes");
    TORCH_CHECK(x.size(1) == y.size(1), "x and y have incompatible shapes");
    TORCH_CHECK(x.size(2) == y.size(2), "x and y have incompatible shapes");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.size(3) % 2 == 0, "x.size(3) must be even");

    size_t numel = x.numel();
    size_t dim = x.size(3);
    size_t blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS);
    mul_softplus_broadcast_kernel_r<<<blocks, NUM_THREADS, 0, stream>>>
    (
        (half*) x.data_ptr(),
        (const half*) y.data_ptr(),
        numel,
        dim
    );

    cuda_check(cudaPeekAtLastError());
}

// Split per-head-interleaved projection output [.., heads, (q: head_dim, g: head_dim)] into
// contiguous q and g tensors (mirrors activation.cu; needed by libtorch/attention.cu on ROCm)

__global__ __launch_bounds__(NUM_THREADS)
void deinterleave_qg_kernel_r
(
    const uint4* __restrict__ qg,
    uint4* __restrict__ q,
    uint4* __restrict__ g,
    const int hd8,                  // head_dim / 8
    const size_t n8                 // rows * heads * head_dim / 8
)
{
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= n8) return;
    size_t d = i % hd8;
    size_t h = i / hd8;
    size_t src = h * 2 * hd8 + d;
    q[i] = qg[src];
    g[i] = qg[src + hd8];
}

void deinterleave_qg_gr
(
    const at::Tensor& qg,           // (.., heads * 2 * head_dim) half
    at::Tensor& q,                  // out (.., heads * head_dim) half
    at::Tensor& g,                  // out (.., heads * head_dim) half
    int head_dim,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(qg.device());
    hipStream_t stream = graph ? graph->capture_stream : c10::hip::getCurrentHIPStream(qg.device().index()).stream();

    TORCH_CHECK_DTYPE(qg, kHalf);
    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE(g, kHalf);
    TORCH_CHECK(head_dim % 8 == 0, "head_dim must be a multiple of 8");
    TORCH_CHECK(qg.is_contiguous() && q.is_contiguous() && g.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(q.numel() == g.numel() && q.numel() * 2 == qg.numel(), "size mismatch");

    int hd8 = head_dim / 8;
    size_t n8 = q.numel() / 8;
    size_t blocks = CEIL_DIVIDE(n8, NUM_THREADS);

    deinterleave_qg_kernel_r<<<blocks, NUM_THREADS, 0, stream>>>
    (
        (const uint4*) qg.data_ptr(),
        (uint4*) q.data_ptr(),
        (uint4*) g.data_ptr(),
        hd8,
        n8
    );

    cuda_check(cudaPeekAtLastError());
}

void deinterleave_qg
(
    const at::Tensor& qg,
    at::Tensor& q,
    at::Tensor& g,
    int head_dim
)
{
    deinterleave_qg_gr(qg, q, g, head_dim, nullptr);
}

#endif // USE_ROCM
