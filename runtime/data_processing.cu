#include "data_processing.h"

__global__ void preprocess_grid_gpu(const cuComplex* y, __half* y_realimag,
                                    int batch_size, int num_rx_ant,
                                    int num_ofdm_symbols, int num_subcarriers)
{
    // This kernel needs to map the input layout (e.g., [batch, ant, sym, sub])
    // to the output layout ([batch, sym, sub, ant*2]) and split complex to real/imag.

    int b  = blockIdx.z;
    int t  = blockIdx.y; // time / symbol
    int f  = blockIdx.x; // freq / subcarrier
    int a  = threadIdx.x; // antenna

    if (b < batch_size && t < num_ofdm_symbols && f < num_subcarriers && a < num_rx_ant) {
        // Calculate source index from the input Sionna tensor layout
        int src_idx = b * (num_rx_ant * num_ofdm_symbols * num_subcarriers) +
                      a * (num_ofdm_symbols * num_subcarriers) +
                      t * num_subcarriers +
                      f;

        // Calculate destination indices for the real and imaginary parts
        int dest_idx_real = b * (num_ofdm_symbols * num_subcarriers * num_rx_ant * 2) +
                            t * (num_subcarriers * num_rx_ant * 2) +
                            f * (num_rx_ant * 2) +
                            a;
        int dest_idx_imag = dest_idx_real + num_rx_ant;

        cuComplex val = y[src_idx];
        y_realimag[dest_idx_real] = __float2half(val.x); // Real part
        y_realimag[dest_idx_imag] = __float2half(val.y); // Imaginary part
    }
}

__global__ void postprocess_llr_gpu(const __half* llr_in, int16_t* llr_out, size_t num_llrs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_llrs) {
        // OAI uses a scaling factor of 2^8 for int16 LLRs
        const float scale = 256.0f;
        float llr_f32 = __half2float(llr_in[idx]);
        // Saturate to int16 range if necessary
        float scaled_val = roundf(llr_f32 * scale);
        if (scaled_val > 32767.0f) scaled_val = 32767.0f;
        if (scaled_val < -32768.0f) scaled_val = -32768.0f;
        llr_out[idx] = (int16_t)scaled_val;
    }
}


// --- Wrapper Functions ---

void preprocess_grid_kernel(const cuComplex* d_in, __half* d_out, int batch_size,
                             int num_rx_ant, int num_ofdm_symbols, int num_subcarriers,
                             cudaStream_t stream)
{
    dim3 grid_dim(num_subcarriers, num_ofdm_symbols, batch_size);
    dim3 block_dim(num_rx_ant);
    preprocess_grid_gpu<<<grid_dim, block_dim, 0, stream>>>(d_in, d_out, batch_size, num_rx_ant, num_ofdm_symbols, num_subcarriers);
}

void postprocess_llr_kernel(const __half* d_in, int16_t* d_out, size_t num_llrs,
                             cudaStream_t stream)
{
    dim3 grid_dim((num_llrs + 255) / 256);
    dim3 block_dim(256);
    postprocess_llr_gpu<<<grid_dim, block_dim, 0, stream>>>(d_in, d_out, num_llrs);
}