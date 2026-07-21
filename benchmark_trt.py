import tensorrt as trt
import torch
import time
import os

def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # TensorRT 10.x 이상부터는 명시적 배치가 기본값이므로 인자 생략
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    # TensorRT 11+ 에서는 기본적으로 가용한 최적의 정밀도(TF32 등)를 사용합니다.
    print("기본 정밀도(TF32 등)로 엔진을 빌드합니다.")
        
    # 워크스페이스 메모리 할당 (TensorRT 8.4 이상 권장 방식: set_memory_pool_limit)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024) # 2GB
        
    print(f"ONNX 파일 파싱 중... ({onnx_file_path})")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ONNX 파싱 실패:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    print("TensorRT 엔진 빌드 중... (몇 분 정도 소요될 수 있습니다)")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("엔진 빌드에 실패했습니다.")
        return None
        
    with open(engine_file_path, 'wb') as f:
        f.write(engine_bytes)
    print(f"엔진 저장 완료: {engine_file_path}")
    
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_bytes)

def benchmark_engine(engine):
    print("추론 컨텍스트 생성 중...")
    context = engine.create_execution_context()
    
    # TensorRT I/O 텐서 바인딩 정보 파악
    num_io_tensors = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(num_io_tensors)]
    
    input_name = tensor_names[0]
    input_shape = engine.get_tensor_shape(input_name)
    output_names = tensor_names[1:]
    
    # PyTorch를 이용하여 GPU(CUDA) 메모리 할당
    # TensorRT V3 API는 메모리 포인터(data_ptr)를 직접 연결하여 통신 오버헤드를 극소화합니다.
    dummy_input = torch.randn(tuple(input_shape), dtype=torch.float32, device='cuda')
    
    outputs = []
    for name in output_names:
        shape = engine.get_tensor_shape(name)
        outputs.append(torch.empty(tuple(shape), dtype=torch.float32, device='cuda'))
        
    # 텐서 메모리 주소 바인딩
    context.set_tensor_address(input_name, dummy_input.data_ptr())
    for name, out_tensor in zip(output_names, outputs):
        context.set_tensor_address(name, out_tensor.data_ptr())
        
    # PyTorch 현재 CUDA 스트림 가져오기
    stream = torch.cuda.current_stream().cuda_stream
    
    print("엔진 웜업(Warm-up) 중...")
    for _ in range(50):
        context.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()
    
    print("본격적인 성능(FPS) 벤치마킹 시작 (1000 프레임)...")
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
    print(f"✨ TensorRT 달성 FPS: {fps:.2f} FPS ✨")
    print("=" * 50)

if __name__ == '__main__':
    onnx_path = "checkpoints/best_model.onnx"
    engine_path = "checkpoints/best_model.engine"
    
    if not os.path.exists(engine_path):
        engine = build_engine(onnx_path, engine_path)
    else:
        print(f"기존 TensorRT 엔진 로드 중: {engine_path}")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            
    if engine is not None:
        benchmark_engine(engine)
