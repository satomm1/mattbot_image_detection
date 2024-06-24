import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load the TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Prepare input data
input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)  # Example input shape

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def infer(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()
    return h_output

# Load the TensorRT engine
engine = load_engine("yolov8n.engine")

# # Allocate buffers
# h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)

# # Create execution context
# context = engine.create_execution_context()

# # Copy input data to host input buffer
# np.copyto(h_input, input_data.ravel())

# # Perform inference
# output = infer(context, h_input, d_input, h_output, d_output, stream)

# print("Inference output:", output)