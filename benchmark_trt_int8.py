import os
import glob
import cv2
import numpy as np
import time
import torch
import tensorrt as trt
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

class SUNSEGDataReader(CalibrationDataReader):
    def __init__(self, data_dir, input_name, batch_size=1, num_samples=100):
        super().__init__()
        self.input_name = input_name
        self.batch_size = batch_size
        self.enum_data = iter(self._get_batches(data_dir, num_samples))
        
    def _get_batches(self, data_dir, num_samples):
        search_pattern = os.path.join(data_dir, "**", "Frame", "**", "*.jpg")
        image_paths = glob.glob(search_pattern, recursive=True)
        image_paths += glob.glob(os.path.join(data_dir, "**", "Frame", "**", "*.png"), recursive=True)
        
        np.random.seed(42)
        np.random.shuffle(image_paths)
        image_paths = image_paths[:num_samples]
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_data = np.zeros((len(batch_paths), 3, 352, 352), dtype=np.float32)
            for j, path in enumerate(batch_paths):
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (352, 352))
                    img = img.astype(np.float32) / 255.0
                    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                    img = np.transpose(img, (2, 0, 1))
                    batch_data[j] = img
            yield {self.input_name: batch_data}

    def get_next(self):
        try:
            return next(self.enum_data)
        except StopIteration:
            return None

def quantize_onnx_to_int8(input_model_path, output_model_path, data_dir):
    print("ONNXRuntime을 이용한 INT8 양자화(PTQ) 시작...")
    # 입력 이름 추출
    model = onnx.load(input_model_path)
    input_name = model.graph.input[0].name
    
    data_reader = SUNSEGDataReader(data_dir, input_name=input_name, batch_size=1, num_samples=50)
    
    quantize_static(
        model_input=input_model_path,
        model_output=output_model_path,
        calibration_data_reader=data_reader,
        quant_format=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        extra_options={'ActivationSymmetric': True, 'WeightSymmetric': True}
    )
    print(f"INT8 양자화 모델 저장 완료: {output_model_path}")

def build_engine_from_qdq(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    print("기본 정밀도로 엔진을 빌드합니다. (QDQ 노드가 있으면 자동으로 INT8 가속 적용)")
    
    print(f"ONNX(QDQ) 파일 파싱 중... ({onnx_file_path})")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    print("TensorRT 엔진 빌드 중...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        return None
        
    with open(engine_file_path, 'wb') as f:
        f.write(engine_bytes)
    print(f"엔진 저장 완료: {engine_file_path}")
    
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_bytes)

def benchmark_engine(engine):
    context = engine.create_execution_context()
    num_io_tensors = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(num_io_tensors)]
    
    input_name = tensor_names[0]
    input_shape = engine.get_tensor_shape(input_name)
    output_names = tensor_names[1:]
    
    dummy_input = torch.randn(tuple(input_shape), dtype=torch.float32, device='cuda')
    
    outputs = []
    for name in output_names:
        shape = engine.get_tensor_shape(name)
        outputs.append(torch.empty(tuple(shape), dtype=torch.float32, device='cuda'))
        
    context.set_tensor_address(input_name, dummy_input.data_ptr())
    for name, out_tensor in zip(output_names, outputs):
        context.set_tensor_address(name, out_tensor.data_ptr())
        
    stream = torch.cuda.current_stream().cuda_stream
    
    print("엔진 웜업(Warm-up) 중...")
    for _ in range(50):
        context.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()
    
    print("INT8本格적인 성능(FPS) 벤치마킹 시작 (1000 프레임)...")
    num_iters = 1000
    start_time = time.time()
    for _ in range(num_iters):
        context.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_iters / total_time
    ms_per_frame = (total_time / num_iters) * 1000
    
    print("\n" + "=" * 50)
    print(f"총 {num_iters} 프레임 처리 시간: {total_time:.2f} 초")
    print(f"단일 프레임 평균 처리 시간: {ms_per_frame:.2f} ms")
    print(f"[INT8] TensorRT 달성 FPS: {fps:.2f} FPS")
    print("=" * 50)

if __name__ == '__main__':
    onnx_path = "checkpoints/best_model.onnx"
    int8_onnx_path = "checkpoints/best_model_int8.onnx"
    engine_path = "checkpoints/best_model_int8.engine"
    data_dir = "E:/Task_DE-ColonSegNet/datasets/SUN-SEG_dataset"
    
    if not os.path.exists(int8_onnx_path):
        quantize_onnx_to_int8(onnx_path, int8_onnx_path, data_dir)
        
    if not os.path.exists(engine_path):
        engine = build_engine_from_qdq(int8_onnx_path, engine_path)
    else:
        print(f"기존 INT8 TensorRT 엔진 로드 중: {engine_path}")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            
    if engine is not None:
        benchmark_engine(engine)
