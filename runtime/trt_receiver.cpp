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

// This is the primary user-facing function that replaces trt_demapper_decode.
// It handles memory copies and preprocessing before calling the internal run function.
extern "C" void trt_receiver_forward(TRTContext* context_, cudaStream_t stream,
                                     const cuComplex* y_host,
                                     const float* no_host,
                                     size_t batch_size,
                                     int16_t* llrs_host)
{
    auto& context = context_ ? *context_ : trt_receiver_init_context(0);
    if (context_ && stream == 0)
        stream = context.default_stream;

    // --- 1. PREPARATION: Move data to GPU and preprocess ---
    cuComplex* y_device = nullptr;
    size_t y_elements = batch_size * NUM_RX_ANT * NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS;
    CHECK_CUDA(cudaMalloc((void**)&y_device, y_elements * sizeof(cuComplex)));
    CHECK_CUDA(cudaMemcpyAsync(y_device, y_host, y_elements * sizeof(cuComplex), cudaMemcpyHostToDevice, stream));

    // KERNEL 1: Preprocess the grid on the GPU
    // This call needs to be implemented in data_processing.cu
    preprocess_grid_kernel(y_device, context.y_realimag_buffer, batch_size,
                           NUM_RX_ANT, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS,
                           stream);

    CHECK_CUDA(cudaMemcpyAsync(context.no_buffer, no_host, batch_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    // --- 2. INFERENCE ---
    // The trt_receiver_run function now launches the graph if available, or enqueues otherwise.
    // We can simplify this by just launching the graph directly here.
#ifdef USE_GRAPHS
    if (context.graph_exec) {
        // Update the input shapes for this specific batch size before launching the graph
        context.trt->setInputShape("y_realimag", Dims4(batch_size, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS, 2 * NUM_RX_ANT));
        context.trt->setInputShape("no", Dims1(batch_size));
        cudaGraphLaunch(context.graph_exec, stream);
    } else {
        // Fallback if graphs are not used or failed to instantiate
        trt_receiver_run(&context, stream, context.y_realimag_buffer, context.no_buffer, batch_size, context.output_llr_buffer);
    }
#else
    trt_receiver_run(&context, stream, context.y_realimag_buffer, context.no_buffer, batch_size, context.output_llr_buffer);
#endif

    // --- 3. GET RESULTS: Postprocess and move data to CPU ---
    size_t llr_elements = batch_size * NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS * BITS_PER_SYMBOL;
    int16_t* llr_output_target = context.host_llr_buffer ? context.host_llr_buffer : llrs_host;

    // KERNEL 2: Postprocess the LLRs on the GPU, writing to the target buffer
    // This call needs to be implemented in data_processing.cu
    postprocess_llr_kernel(context.output_llr_buffer, llr_output_target, llr_elements, stream);

    // If we used a persistent buffer, we need to copy the final result back to the caller's buffer.
    // If not, the kernel wrote directly to it.
#if defined(PERSISTENT_DEVICE_MEMORY) && !defined(USE_UNIFIED_MEMORY)
    CHECK_CUDA(cudaMemcpyAsync(llrs_host, llr_output_target, llr_elements * sizeof(int16_t), cudaMemcpyDeviceToHost, stream));
#endif

    // --- 4. CLEANUP ---
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaFree(y_device));

#if defined(PERSISTENT_DEVICE_MEMORY) && defined(USE_UNIFIED_MEMORY)
    // If using unified memory, the host can see the results after sync
    if (context.host_llr_buffer) {
        memcpy(llrs_host, context.host_llr_buffer, llr_elements * sizeof(int16_t));
    }
#endif
}


TRTContext& trt_receiver_init_context(int make_stream) {
    if (!thread_context)
        thread_context = new TRTContext();
    auto& context = *thread_context;
    if (context.trt) // lazy
        return context;

    printf("Initializing TRT receiver context (TID %d)\n", (int) gettid());
#if NV_TENSORRT_MAJOR >= 10
    context.trt = engine->createExecutionContext(ExecutionContextAllocationStrategy::kSTATIC);
#else
    context.trt = engine->createExecutionContextWithoutDeviceMemory();
    size_t preallocSize = engine->getDeviceMemorySize();
    cudaMalloc(&context.prealloc_memory, preallocSize);
    if (0 == preallocStatus && context.prealloc_memory)
        context.trt->setDeviceMemory(context.prealloc_memory);
#endif

    if (make_stream)
        CHECK_CUDA(cudaStreamCreateWithFlags(&context.default_stream, cudaStreamNonBlocking));

    // ---- 1. Allocate GPU Buffers ----
    size_t y_size = MAX_BATCH_SIZE * NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS * 2 * NUM_RX_ANT;
    size_t no_size = MAX_BATCH_SIZE;
    size_t out_size = MAX_BATCH_SIZE * NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS * BITS_PER_SYMBOL;

    cudaMalloc((void**) &context.y_realimag_buffer, sizeof(__half) * y_size);
    cudaMalloc((void**) &context.no_buffer, sizeof(float) * no_size);
    cudaMalloc((void**) &context.output_llr_buffer, sizeof(__half) * out_size);

#ifdef PERSISTENT_DEVICE_MEMORY
    #ifdef USE_UNIFIED_MEMORY
        #define DEVICE_IO_ALLOC cudaHostAlloc
    #else
        #define DEVICE_IO_ALLOC(p, s, f) cudaMalloc(p, s)
    #endif
    // Allocate a persistent, host-mapped buffer for the final output
    DEVICE_IO_ALLOC((void**) &context.host_llr_buffer, sizeof(*context.host_llr_buffer) * out_size, cudaHostAllocMapped);
    #undef DEVICE_IO_ALLOC
#endif


// --- 2. Record CUDA Graph ----
// START marker-record-graph
#ifdef USE_GRAPHS
    // We record a single graph that contains the main inference call.
    // The pre/post-processing kernels that move data to/from these buffers
    // will be called outside the graph in the `forward` function.

    cudaStream_t stream = context.default_stream;

    // Set the tensor addresses to our persistent device buffers *before* capturing.
    context.trt->setTensorAddress("y_realimag", context.y_realimag_buffer);
    context.trt->setTensorAddress("no", context.no_buffer);
    context.trt->setTensorAddress("output_1", context.output_llr_buffer);

    // Set shapes for the maximum batch size during capture.
    // These will be updated at runtime for each specific batch.
    context.trt->setInputShape("y_realimag", Dims4(MAX_BATCH_SIZE, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS, 2 * NUM_RX_ANT));
    context.trt->setInputShape("no", Dims1(MAX_BATCH_SIZE));

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // The only operation needed inside the graph is the inference itself.
    context.trt->enqueueV3(stream);

    CHECK_CUDA(cudaStreamEndCapture(stream, &context.graph));
    CHECK_CUDA(cudaGraphInstantiate(&context.graph_exec, context.graph, 0));
    PRINT_INFO_VERBOSE("Recorded and instantiated CUDA graph for receiver (TID %d)\n", (int) gettid());
#endif
// END marker-record-graph

    // keep track of active thread contexts for shutdown
    TRTContext* self = &context;
    __atomic_exchange(&initialized_thread_contexts, &self, &self->next_initialized_context, __ATOMIC_ACQ_REL);

    return context;
}

// Helper to read the .plan engine file from disk
std::vector<char> readModelFromFile(char const* filepath) {
    std::vector<char> bytes;
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        logger.log(Logger::Severity::kERROR, filepath);
        return bytes;
    }
    fseek(f, 0, SEEK_END);
    bytes.resize((size_t) ftell(f));
    fseek(f, 0, SEEK_SET);
    if (bytes.size() != fread(bytes.data(), 1, bytes.size(), f))
        logger.log(Logger::Severity::kWARNING, filepath);
    fclose(f);
    return bytes;
}

extern "C" void trt_receiver_shutdown();

// Global one-time initialization
extern "C" TRTContext* trt_receiver_init(int make_stream) {
    if (runtime)  // lazy, global init
        return &trt_receiver_init_context(make_stream);

    printf("Initializing TRT runtime (TID %d)\n", (int) gettid());
    runtime = createInferRuntime(logger);
    printf("Loading TRT engine from: %s\n", trt_weight_file);
    std::vector<char> modelData = readModelFromFile(trt_weight_file);
    engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());

#ifdef ENABLE_NANOBIND
    // This ensures shutdown is called automatically when the Python module unloads
    static struct AutoTRTShutdown {
        ~AutoTRTShutdown() {
            trt_receiver_shutdown();
        }
    } trt_shutdown_guard;
#endif

    return &trt_receiver_init_context(make_stream);
}

// Global one-time shutdown
extern "C" void trt_receiver_shutdown() {
    TRTContext* active_context = nullptr;
    __atomic_exchange(&initialized_thread_contexts, &active_context, &active_context, __ATOMIC_ACQ_REL);
    while (active_context) {
#ifdef USE_GRAPHS
        if (active_context->graph_exec) cudaGraphExecDestroy(active_context->graph_exec);
        if (active_context->graph) cudaGraphDestroy(active_context->graph);
#endif
#if NV_TENSORRT_MAJOR >= 10
        delete active_context->trt;
#else
        if(active_context->trt) active_context->trt->destroy();
#endif

        cudaFree(active_context->prealloc_memory);
        // Free the new receiver buffers
        cudaFree(active_context->y_realimag_buffer);
        cudaFree(active_context->no_buffer);
        cudaFree(active_context->output_llr_buffer);

#ifdef PERSISTENT_DEVICE_MEMORY
        cudaFree(active_context->host_llr_buffer);
#endif
        if (active_context->default_stream)
            cudaStreamDestroy(active_context->default_stream);

        TRTContext* next_context = active_context->next_initialized_context;
        delete active_context;
        active_context = next_context;
    }

#if NV_TENSORRT_MAJOR >= 10
    delete engine;
    delete runtime;
#else
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
#endif
    engine = nullptr;
    runtime = nullptr;
    printf("TRT receiver shutdown complete.\n");
}

#ifdef ENABLE_NANOBIND

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <complex> // Needed for std::complex

namespace nb = nanobind;

// This tells nanobind how to handle the __half type if you pass it to/from Python.
namespace nanobind::detail {
    template <> struct dtype_traits<__half> {
        static constexpr dlpack::dtype value {
            (uint8_t) dlpack::dtype_code::Float, 16, 1
        };
        static constexpr auto name = const_name("float16");
    };
}

// Rename the module to trt_receiver
NB_MODULE(trt_receiver, m) {
    // Define a single function "forward" to be called from Python
    m.def("forward", [](const nb::ndarray<std::complex<float>, nb::shape<-1, -1, -1, -1>, nb::device::cpu>& y,
                           const nb::ndarray<float, nb::shape<-1>, nb::device::cpu>& no) {
        
        auto* context = trt_receiver_init(1); 
        size_t batch_size = y.shape(0);
        
        size_t num_llrs = batch_size * NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS * BITS_PER_SYMBOL;
        
        int16_t *data = new int16_t[num_llrs];
        memset(data, 0, sizeof(*data) * num_llrs);
        nb::capsule owner(data, [](void *p) noexcept { delete[] (int16_t*) p; });

        trt_receiver_forward(context, 0, 
                             (const cuComplex*)y.data(), // Cast std::complex<float>* to cuComplex*
                             no.data(), 
                             batch_size,
                             data);

        // Define the shape of the output NumPy array
        // [batch, num_rx, num_re, bits_per_symbol]
        size_t shape[4] = {batch_size, NUM_RX_ANT, NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS, BITS_PER_SYMBOL};

        return nb::ndarray<nb::numpy, int16_t, nb::ndim<4>>(data, shape, owner);
    });
}

#endif