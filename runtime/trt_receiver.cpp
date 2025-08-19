#include "NvInfer.h"
#include "data_processing.h"
#include <cuda_fp16.h>
#include <cuComplex.h> // new include needed because rx has complex tensors
#include <vector>
#include <cstdio>
#include <unistd.h>
#include <time.h>

using namespace nvinfer1;

static IRuntime* runtime = nullptr;
static ICudaEngine* engine = nullptr;

static uint32_t const NUM_OFDM_SYMBOLS = 14;
static uint32_t const NUM_SUBCARRIERS = 76;
static uint32_t const NUM_RX_ANT = 2;
static uint32_t const BITS_PER_SYMBOL = 2;
static uint32_t const MAX_BATCH_SIZE = 512;

#define PERSISTENT_DEVICE_MEMORY
#define USE_UNIFIED_MEMORY
#define USE_GRAPHS

struct TRTContext {
    cudaStream_t default_stream = 0;
    IExecutionContext* trt = nullptr;
    void* prealloc_memory = nullptr;
    cudaGraph_t graph = nullptr;  
    cudaGraphExec_t graph_exec = nullptr;

    __half* y_realimag_buffer = nullptr;
    float* no_buffer = nullptr;

    __half* output_llr_buffer = nullptr;

    #ifdef PERSISTENT_DEVICE_MEMORY
    int16_t* host_llr_buffer = nullptr;
    #endif

    // list of thread contexts for shutdown
    TRTContext* next_initialized_context = nullptr;
};

static __thread TRTContext* thread_context = nullptr;
static TRTContext* initialized_thread_contexts = nullptr;

#ifdef PRINT_TIMES
struct TimeMeasurements {
    unsigned long long total_ns;
    unsigned long long max_ns;
    unsigned count;
};
static __thread struct TimeMeasurements cuda_time;

static unsigned add_measurement(struct TimeMeasurements& time, unsigned long long time_ns) {
    time.total_ns += time_ns;
    if (time_ns > time.max_ns)
        time.max_ns = time_ns;
    return time.count++;
}
#endif

#define CHECK_CUDA(call) do { cudaError_t err = (call); \
    if (err) printf("CUDA error %d: %s; in %s|%d|: %s\n", (int) err, cudaGetErrorString(err), __FILE__, __LINE__, #call); } while (false)

struct Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            printf("TensorRT %s: %s\n", severity == Severity::kWARNING ? "WARNING" : "ERROR", msg);
    }
} logger;

static char const* trt_weight_file = "model_neuralrx.plan";

extern "C" void trt_receiver_configure(char const* weight_file) {
    trt_weight_file = weight_file;
}

TRTContext& trt_receiver_init_context(int make_stream);

// new core function for a single inference run pass
void trt_receiver_run(TRTContext* context_, cudaStream_t stream,
                        const __half* y_realimag_device, const float* no_device,
                        size_t batch_size, __half* llr_output_device)
{
    auto& context = context_ ? *context_ : trt_receiver_init_context(0);
    if (context_ && stream == 0)
        stream = context.default_stream;

    // Bind the device pointers to the named inputs and outputs of the engine.
    // Changed the names for the the inputs on new reciever ONNX model
    context.trt->setTensorAddress("y_realimag", (void*)y_realimag_device);
    context.trt->setTensorAddress("no", (void*)no_device);
    context.trt->setTensorAddress("output_1", (void*)llr_output_device); // "output_1" is a common default name

    // Set the input shapes for this specific batch like done before for the demapper, change to new sizes and names
    context.trt->setInputShape("y_realimag", Dims4(batch_size, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS, 2 * NUM_RX_ANT));
    context.trt->setInputShape("no", Dims1(batch_size));

    // Execute the async inference by putting into GPU queue
    context.trt->enqueueV3(stream);
}

// Forward declaration for the main function we will build in the next step.
// This function will handle preprocessing and call trt_receiver_run.
extern "C" void trt_receiver_forward(TRTContext* context_, cudaStream_t stream,
                                     /* inputs */ const cuComplex* y_host, const float* no_host,
                                     size_t batch_size,
                                     /* outputs */ int16_t* llrs_host);
