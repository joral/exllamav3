// ROCm fallback for the EXL3 GEMM kernels (exl3_gemm / exl3_mgemm and their graph variants).
//
// The CUDA kernels are a hand-written MMA/cp.async implementation with no RDNA equivalent, so
// this TU reimplements the same composition from bound primitives, all proven numerically
// equivalent to exl3_gemv on gfx1200 (rel err ~8e-4 = fp16 noise):
//
//   y = hgemm_recon(x, reconstruct_had_slice(trellis, suh, svh))
//
// reconstruct_had_slice emits ORIGINAL-basis weights (both Hadamards + sign vectors + scales
// folded in): W = diag(suh) . H128 . W_hat . H128 . diag(svh). The gemm therefore runs on the
// RAW input — no input-side Hadamard staging (an earlier version staged x through had_r_128,
// double-applying suh/H and corrupting every output; verified against exl3_gemv, rel ~1.7e-3).
// hgemm_recon degrades to cuBLAS with fp32 accumulation.
//
// Graph capture notes:
// - Every call is a plain kernel launch or a cuBLAS gemm over the static DevCtx workspace:
//   no allocations after the first (eager) call, whose weight scratch is cached statically.
// - All launches are routed onto the capture stream by temporarily swapping the ambient
//   CUDA stream while the graph variants run, so aten-bound primitives record correctly.
// - The pointer tables (B/suh/svh/c_ptrs) are device tensors of int64. Their values are
//   synced to a host mirror once per table (on the eager pass, when syncs are legal), so the
//   captured pass never syncs and replays never revisit this code at all.
// - Patchable input: the caller stages x into a static buffer (BC_Attention does this on
//   ROCm), since aten/cublas nodes have no patchable kernel-argument sites.

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Exceptions.h>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdint>

#include "exl3_gemm.cuh"
#include "exl3_gemv.cuh"
#include "exl3_gemv_int8.cuh"
#include "exl3_devctx.cuh"
#include "reconstruct.cuh"
#include "hadamard.cuh"
#include "../hgemm.cuh"
#include "../dsa_topk.cuh"

// ---------------------------------------------------------------------------
// Static caches. Keyed by raw table pointer (tables are build-once objects owned by
// MultiLinear/SlicedMultiLinear/BC_Attention and outlive every graph that references them).
// ---------------------------------------------------------------------------

namespace
{
    std::mutex shim_mutex;

    // device table data_ptr -> host copy of the int64 entries
    std::unordered_map<void*, std::vector<int64_t>> table_mirror;

    // (device index, k, n) -> reconstructed-weight scratch (kept alive for address stability)
    std::unordered_map<int64_t, at::Tensor> weight_scratch;

    const int64_t* mirror_entries(const at::Tensor& table)
    {
        void* key = table.data_ptr();
        auto it = table_mirror.find(key);
        if (it == table_mirror.end())
        {
            at::Tensor host = table.to(at::kCPU).contiguous();
            int64_t* d = (int64_t*) host.data_ptr();
            it = table_mirror.emplace(key, std::vector<int64_t>(d, d + table.numel())).first;
        }
        return it->second.data();
    }

    // Cached host mirror of a device int32 list. Safe for capture: the mirror is built on the
    // eager pass and reused by pointer identity; the lists are static per slot
    std::unordered_map<void*, std::vector<int>> int_mirror;
    const int* mirror_ints(const at::Tensor& table)
    {
        void* key = table.data_ptr();
        auto it = int_mirror.find(key);
        if (it == int_mirror.end())
        {
            TORCH_CHECK(table.scalar_type() == at::kInt, "mirror_ints: int32 expected");
            at::Tensor host = table.to(at::kCPU).contiguous();
            const int* d = (const int*) host.data_ptr();
            it = int_mirror.emplace(key, std::vector<int>(d, d + table.numel())).first;
        }
        return it->second.data();
    }

    at::Tensor& weight_scratch_for(int device, int64_t k, int64_t n)
    {
        int64_t key = ((int64_t) device << 50) ^ (k << 25) ^ n;
        auto it = weight_scratch.find(key);
        if (it == weight_scratch.end())
        {
            at::Tensor t = at::empty(
                {k, n},
                at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, device));
            it = weight_scratch.emplace(key, t).first;
        }
        return it->second;
    }

    // Input-side Hadamard workspace for the GEMV route, (rows, k) fp16; allocated on the
    // eager pass and reused by address afterwards (capture-safe)
    std::unordered_map<int64_t, at::Tensor> gemv_had;
    at::Tensor& gemv_had_for(int device, int64_t rows, int64_t k)
    {
        int64_t key = ((int64_t) device << 50) ^ (rows << 25) ^ k;
        auto it = gemv_had.find(key);
        if (it == gemv_had.end())
        {
            at::Tensor t = at::empty(
                {rows, k},
                at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, device));
            it = gemv_had.emplace(key, t).first;
        }
        return it->second;
    }

    // Contiguous scratch for strided-output GEMV calls, (rows, n) fp16 or fp32; allocated on
    // the eager pass, stable address afterwards (capture-safe)
    std::unordered_map<int64_t, at::Tensor> gemv_c_scratch;
    at::Tensor& gemv_c_scratch_for(int device, int64_t rows, int64_t n, bool fp32)
    {
        int64_t key = ((int64_t) device << 52) ^ (rows << 26) ^ n ^ (fp32 ? (1LL << 51) : 0);
        auto it = gemv_c_scratch.find(key);
        if (it == gemv_c_scratch.end())
        {
            at::Tensor t = at::empty(
                {rows, n},
                at::TensorOptions().dtype(fp32 ? at::kFloat : at::kHalf).device(at::kCUDA, device));
            it = gemv_c_scratch.emplace(key, t).first;
        }
        return it->second;
    }

    // Route all aten-bound launches to the capture stream while a graph variant runs
    struct StreamOverride
    {
        c10::optional<c10::cuda::CUDAStream> prev;
        StreamOverride(Graph* graph)
        {
            if (graph)
            {
                prev = c10::cuda::getCurrentCUDAStream();
                c10::cuda::setCurrentCUDAStream(
                    c10::cuda::getStreamFromExternal(graph->capture_stream, c10::cuda::current_device()));
            }
        }
        ~StreamOverride()
        {
            if (prev.has_value())
                c10::cuda::setCurrentCUDAStream(prev.value());
        }
    };

    at::Tensor ptr_tensor(void* ptr, at::IntArrayRef sizes, at::IntArrayRef strides, at::Device device, at::ScalarType dtype)
    {
        return at::from_blob(ptr, sizes, strides, [](void*) {}, at::TensorOptions().dtype(dtype).device(device));
    }
}

// ---------------------------------------------------------------------------
// Core composition, shared by all entry points.
// ---------------------------------------------------------------------------

namespace
{
    // One slice/matrix: C_j = gemm(A_src, reconstruct_had_slice(B_j)). A_src is the RAW input
    // (rows, k) — all scales and Hadamards live inside the reconstructed weights.
    void gemm_slice(
        const at::Tensor& A_src,          // (rows, k) raw input rows for this source
        const at::Tensor& suh_src,        // (k,) input-side scale, from_blob over table entry
        void* trellis_ptr,                // pre-offset trellis slice start
        void* svh_ptr,                    // pre-offset output-side scale start
        int64_t n_j,                      // slice width
        int64_t c_stride,                 // output row stride (>= n_j)
        void* c_ptr,                      // output slice start
        int64_t rows,
        int K, bool mcg, bool mul1,
        bool has_su_sv,                   // real scales present (not zero-filled fallback)
        at::ScalarType c_dtype,           // caller's output dtype (kHalf or kFloat)
        Graph* graph)
    {
        const at::cuda::OptionalCUDAGuard device_guard(A_src.device());
        int device = (int) A_src.device().index();
        int64_t k = A_src.size(1);

        // GEMV route for decode-sized batches (rows <= 16 on ROCm): the trellis is dequantized
        // in the kernel (no W materialization per call) and supports fp32 output natively.
        // Requires real su/svh, contiguous A and a contiguous output slice (c_stride == n_j).
        // Same kernel-argument layout as the native exl3_gemm, so graph recording restores
        // the patchable A/B/C/scale sites the reconstruct path lacked
        static const bool shim_gemv = []
        {
            const char* e = getenv("EXL3_SHIM_GEMV");
            return !e || e[0] != '0';
        }();
        if (shim_gemv &&
            has_su_sv && rows <= 16 && k % 128 == 0 && n_j % 128 == 0 &&
            K >= 2 && K <= 6 &&
            A_src.is_contiguous() && suh_src.is_contiguous())
        {
            const bool strided_out = (c_stride != n_j);
            const bool c_fp32 = (c_dtype == at::kFloat);

            // Output: direct into the caller's slice when contiguous; otherwise a contiguous
            // scratch followed by a strided D2D copy (stream-ordered, capture-safe)
            at::Tensor gemv_c_t = strided_out
                ? gemv_c_scratch_for(device, rows, n_j, c_fp32)
                : at::Tensor();
            void* gemv_C = strided_out ? gemv_c_t.data_ptr() : c_ptr;

            at::Tensor& A_had = gemv_had_for(device, rows, k);
            had_r_128(A_src, A_had, suh_src, c10::nullopt, 1.0f);

            const half* gemv_A = (const half*) A_had.data_ptr();
            const uint16_t* gemv_B = (const uint16_t*) trellis_ptr;
            const half* gemv_suh = (const half*) suh_src.data_ptr();
            half* gemv_Ahad = (half*) A_had.data_ptr();
            const half* gemv_svh = (const half*) svh_ptr;
            int* locks = DevCtx::instance().get_locks(device);
            int size_m = (int) rows;
            int size_k = (int) k;
            int size_n = (int) n_j;
            int cb = mcg ? 1 : (mul1 ? 2 : 0);
            void* kernel_args[] =
            {
                (void*) &gemv_A, (void*) &gemv_B, (void*) &gemv_C,
                (void*) &size_m, (void*) &size_k, (void*) &size_n,
                (void*) &locks, (void*) &gemv_suh, (void*) &gemv_Ahad, (void*) &gemv_svh
            };
            hipStream_t gemv_stream = at::cuda::getCurrentCUDAStream().stream();
            void* gemv_kernel = nullptr;
            if (exl3_gemv_try_launch(
                    kernel_args, size_m, size_k, size_n, K, cb, c_fp32,
                    true, device, gemv_stream, &gemv_kernel, true))
            {
                // Output-side svh Hadamard, in place (scratch or the caller's slice)
                at::Tensor Cout = strided_out
                    ? gemv_c_t
                    : ptr_tensor(c_ptr, {rows, n_j}, {c_stride, 1}, A_src.device(),
                                 c_fp32 ? at::kFloat : at::kHalf);
                had_r_128(Cout, Cout, c10::nullopt,
                          ptr_tensor(svh_ptr, {n_j}, {1}, A_src.device(), at::kHalf), 1.0f);
                if (strided_out)
                {
                    // Scatter rows into the strided carrier
                    size_t elem = c_fp32 ? sizeof(float) : sizeof(__half);
                    hipError_t err = hipMemcpy2DAsync(
                        c_ptr, (size_t) c_stride * elem,
                        gemv_C, (size_t) n_j * elem,
                        (size_t) n_j * elem, (size_t) rows,
                        hipMemcpyDeviceToDevice, gemv_stream);
                    TORCH_CHECK(err == hipSuccess, "gemv route: hipMemcpy2DAsync failed: ",
                                hipGetErrorString(err));
                }
                if (graph)
                {
                    graph->record_param(gemv_kernel, GP_gemm_A, 0);
                    graph->record_param(gemv_kernel, GP_gemm_B_trellis, 1);
                    graph->record_param(gemv_kernel, GP_gemm_C, 2);
                    graph->record_param(gemv_kernel, GP_gemm_B_suh, 7);
                    graph->record_param(gemv_kernel, GP_gemm_A_had, 8);
                    graph->record_param(gemv_kernel, GP_gemm_B_svh, 9);
                    graph->record_param(gemv_kernel, GP_end, 0);
                    graph->need_cublas = true;
                }
                return;
            }
            // Declined (unsupported K, no HIP gemv on this device, ...): fall through to the
            // reconstruct path below
        }

        // Reconstructed weight scratch (allocated on the eager pass, stable address after)
        at::Tensor& w = weight_scratch_for(device, k, n_j);
        at::Tensor trellis = ptr_tensor(
            trellis_ptr, {k / 16, n_j / 16, 16 * K}, {n_j / 16 * 16 * K, 16 * K, 1},
            A_src.device(), at::kShort);
        at::Tensor svh = ptr_tensor(svh_ptr, {n_j}, {1}, A_src.device(), at::kHalf);
        reconstruct_had_slice(w, trellis, suh_src, svh, K, mcg, mul1, 0);

        at::Tensor C = ptr_tensor(c_ptr, {rows, n_j}, {c_stride, 1}, A_src.device(), at::kHalf);
        if (c_dtype == at::kFloat)
        {
            // Caller wants fp32 output (e.g. o_proj / lm_head with fp32 out_dtype). The recon
            // gemm computes fp16; compute into scratch and convert so the caller's fp32 buffer
            // is fully written in fp32 bits
            at::Tensor Cf = at::empty({rows, n_j}, A_src.options().dtype(at::kHalf));
            hgemm_recon(A_src, w, Cf);
            C = ptr_tensor(c_ptr, {rows, n_j}, {c_stride, 1}, A_src.device(), at::kFloat);
            C.copy_(Cf);
        }
        else
            hgemm_recon(A_src, w, C);
        if (graph) graph->need_cublas = true;
    }
}

// ---------------------------------------------------------------------------
// exl3_gemm / exl3_gemm_gr: single matrix. B is the trellis tensor itself.
// ---------------------------------------------------------------------------

namespace
{
    int exl3_gemm_impl(
        const at::Tensor& A,
        const at::Tensor& B,
        at::Tensor& C,
        const c10::optional<at::Tensor>& suh,
        const c10::optional<at::Tensor>& A_had,
        const c10::optional<at::Tensor>& svh,
        bool mcg,
        bool mul1,
        Graph* graph)
    {
        TORCH_CHECK(!graph || A_had.has_value(),
                    "exl3_gemm_gr: A_had staging buffer is required for graph capture");
        at::Tensor A_ = A.view({-1, A.size(-1)});
        at::Tensor C_ = C.view({-1, C.size(-1)});
        int64_t rows = A_.size(0);
        int64_t k = A_.size(1);
        int64_t n = C_.size(1);

        at::Tensor suh_ = suh.has_value()
            ? suh.value()
            : at::empty({k}, A_.options().dtype(at::kHalf));
        at::Tensor svh_ = svh.has_value()
            ? svh.value()
            : at::empty({n}, A_.options().dtype(at::kHalf));

        StreamOverride override_(graph);
        gemm_slice(
            A_, suh_,
            B.data_ptr(), svh_.data_ptr(),
            n, n, C_.data_ptr(),
            rows, (int) (B.size(2) / 16), mcg, mul1,
            suh.has_value() && svh.has_value(),
            C.scalar_type(),
            graph);
        return 0;
    }
}

int exl3_gemm_gr(
    const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx, bool mcg, bool mul1, int force_num_sms,
    Graph* graph)
{
    return exl3_gemm_impl(A, B, C, suh, A_had, svh, mcg, mul1, graph);
}

int exl3_gemm(
    const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx, bool mcg, bool mul1, int force_num_sms)
{
    return exl3_gemm_impl(A, B, C, suh, A_had, svh, mcg, mul1, nullptr);
}

// ---------------------------------------------------------------------------
// exl3_mgemm / exl3_mgemm_gr: pointer-table mode.
//
// Non-sliced (per-matrix, attn.py MultiLinear): B/suh/svh are per-matrix pointer tables with
// full-matrix entries; C is (num_matrices, rows, n). The A_had staging argument is accepted
// for signature compatibility but unused (no input-side Hadamard in this composition).
//
// Sliced (BC_Attention): each entry of B/svh is a pre-offset slice start; size_n_list holds
// the slice widths, n_stride_list the output row strides, c_ptrs the output slice starts and
// had_src_list the source index of each slice (the input Hadamard runs once per source).
// ---------------------------------------------------------------------------

namespace
{
    int exl3_mgemm_impl(
        const at::Tensor& A,
        const at::Tensor& B,
        at::Tensor& C,
        const at::Tensor& suh,
        const at::Tensor& A_had,
        const at::Tensor& svh,
        const c10::optional<at::Tensor>& indices,
        const c10::optional<at::Tensor>& weights,
        int K,
        bool mcg,
        bool mul1,
        int num_tokens,
        const c10::optional<at::Tensor>& size_n_list,
        const c10::optional<at::Tensor>& c_ptrs,
        const c10::optional<at::Tensor>& n_stride_list,
        const c10::optional<at::Tensor>& had_src_list,
        int num_had_src,
        Graph* graph)
    {
        TORCH_CHECK(!indices.has_value() || !indices.value().defined(),
                    "exl3_mgemm ROCm fallback: indices are not supported");
        TORCH_CHECK(!weights.has_value() || !weights.value().defined(),
                    "exl3_mgemm ROCm fallback: weights are not supported");
        TORCH_CHECK(num_tokens == 1,
                    "exl3_mgemm ROCm fallback: only num_tokens == 1 is supported");

        StreamOverride override_(graph);
        const at::cuda::OptionalCUDAGuard device_guard(A.device());

        at::Tensor A_ = A.view({-1, A.size(-1)});
        int64_t rows = A_.size(0);
        int64_t k = A_.size(1);
        TORCH_CHECK(A_.is_contiguous() && A_.dtype() == at::kHalf,
                    "exl3_mgemm ROCm fallback: A must be contiguous fp16");

        std::lock_guard<std::mutex> lock(shim_mutex);

        const int64_t* b_ptrs = mirror_entries(B);
        const int64_t* suh_ptrs = mirror_entries(suh);
        const int64_t* svh_ptrs = mirror_entries(svh);

        bool sliced = size_n_list.has_value() && size_n_list.value().defined() &&
                      size_n_list.value().numel() > 0;

        if (sliced)
        {
            int64_t num_slices = size_n_list.value().numel();
            const int64_t* c_vals = mirror_entries(c_ptrs.value());
            const int* sn = mirror_ints(size_n_list.value());
            const int* st = mirror_ints(n_stride_list.value());
            const int* hs = mirror_ints(had_src_list.value());

            for (int64_t j = 0; j < num_slices; ++j)
            {
                int src = hs[j];
                int64_t n_j = sn[j];
                at::Tensor suh_src = ptr_tensor(
                    (void*) suh_ptrs[src], {k}, {1}, A.device(), at::kHalf);
                gemm_slice(
                    A_, suh_src,
                    (void*) b_ptrs[j], (void*) svh_ptrs[j],
                    n_j, st[j], (void*) c_vals[j],
                    rows, K, mcg, mul1, true, C.scalar_type(), graph);
            }
        }
        else
        {
            int64_t num_matrices = B.numel();
            TORCH_CHECK(C.dim() == 3 && C.size(0) == num_matrices,
                        "exl3_mgemm ROCm fallback: C must be (num_matrices, rows, n)");
            int64_t n = C.size(2);
            at::Tensor C_ = C.view({num_matrices, rows, n});
            for (int64_t e = 0; e < num_matrices; ++e)
            {
                at::Tensor suh_e = ptr_tensor(
                    (void*) suh_ptrs[e], {k}, {1}, A.device(), at::kHalf);
                at::Tensor svh_e = ptr_tensor(
                    (void*) svh_ptrs[e], {n}, {1}, A.device(), at::kHalf);
                gemm_slice(
                    A_, suh_e,
                    (void*) b_ptrs[e], (void*) svh_ptrs[e],
                    n, n, C_.select(0, e).data_ptr(),
                    rows, K, mcg, mul1, true, C.scalar_type(), graph);
            }
        }
        return 0;
    }
}

int exl3_mgemm_gr(
    const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
    const at::Tensor& suh, const at::Tensor& A_had, const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    int K, int force_shape_idx, bool mcg, bool mul1,
    int min_index, int max_index, int force_num_sms,
    Graph* graph,
    int num_tokens,
    const c10::optional<at::Tensor>& size_n_list,
    const c10::optional<at::Tensor>& c_ptrs,
    const c10::optional<at::Tensor>& n_stride_list,
    const c10::optional<at::Tensor>& had_src_list,
    int num_had_src)
{
    TORCH_CHECK(min_index < 0 && max_index < 0,
                "exl3_mgemm_gr ROCm fallback: index bounds are not supported");
    return exl3_mgemm_impl(A, B, C, suh, A_had, svh, indices, weights, K, mcg, mul1,
                           num_tokens, size_n_list, c_ptrs, n_stride_list, had_src_list,
                           num_had_src, graph);
}

int exl3_mgemm(
    const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
    const at::Tensor& suh, const at::Tensor& A_had, const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    int K, int force_shape_idx, uint32_t mcg_mult, uint32_t mul1_mult,
    int min_index, int max_index, int force_num_sms,
    int num_tokens,
    const c10::optional<at::Tensor>& size_n_list,
    const c10::optional<at::Tensor>& c_ptrs,
    const c10::optional<at::Tensor>& n_stride_list,
    const c10::optional<at::Tensor>& had_src_list,
    int num_had_src)
{
    TORCH_CHECK(min_index < 0 && max_index < 0,
                "exl3_mgemm ROCm fallback: index bounds are not supported");
    return exl3_mgemm_impl(A, B, C, suh, A_had, svh, indices, weights, K,
                           mcg_mult != 0, mul1_mult != 0,
                           num_tokens, size_n_list, c_ptrs, n_stride_list, had_src_list,
                           num_had_src, nullptr);
}

// ---------------------------------------------------------------------------
// QSA top-k: not provided on ROCm (the reference model has no sparse indexer). Kept as a
// stub so libtorch/attention.cpp links; a module arming QSA fails loudly here.
// ---------------------------------------------------------------------------

void dsa_topk_gr(
    const at::Tensor& scores,
    at::Tensor& indices,
    int k,
    Graph* graph,
    const c10::optional<at::Tensor>& t_ptr,
    int t_seq)
{
    TORCH_CHECK(false, "dsa_topk_gr: QSA sparse attention is not supported on ROCm");
}

// ---------------------------------------------------------------------------
// Highest K the int8 GEMV path accepts on this arch. Mirrors quant/exl3_gemv_int8.cu, which
// is excluded on ROCm; gfx1200 keeps the conservative default (5) until measured otherwise.
// ---------------------------------------------------------------------------

int exl3_gemv_int8_max_k(int device)
{
    static const int env_max_k = [] { const char* e = getenv("EXL3_INT8_GEMV_MAX_K"); return e ? atoi(e) : 0; }();
    if (env_max_k) return std::min(env_max_k, 8);
    int cc = DevCtx::instance().get_cc(device);
    return (cc == CC_HOPPER || cc == CC_BLACKWELL) ? 6 : 5;
}
