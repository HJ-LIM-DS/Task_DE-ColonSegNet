import tensorrt as trt
import torch
import numpy as np
import time
import os

def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    
    # TensorRT 11+ 에서는 기본적으로 가용한 최적의 정밀도(TF32 등)를 사용합니다.
    # 하지만 QDQ 노드가 있으므로 TRT가 자동으로 INT8을 활성화합니다.
    # 추가로 FP16 폴백을 허용합니다.
    
    # 메모리 풀 제한 설정 (TRT 11.x 권장 방식)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
        
    print(f"ONNX 파일 파싱 중... ({onnx_file_path})")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: ONNX 파싱 실패")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    print("TensorRT INT8(QDQ) 엔진 빌드 중... (수 분 소요될 수 있습니다)")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("엔진 빌드 실패.")
        return None
        
    with open(engine_file_path, 'wb') as f:
        f.write(engine_bytes)
    print(f"엔진 저장 완료: {engine_file_path}")
    
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_bytes)

def benchmark_engine(engine):
    print("추론 컨텍스트 생성 중...")
    context = engine.create_execution_context()
    
    # 입력/출력 텐서 이름 확인 (TRT 10+ 방식)
    num_io_tensors = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(num_io_tensors)]
    
    input_name = tensor_names[0]
    input_shape = engine.get_tensor_shape(input_name)
    output_names = tensor_names[1:]
    
    # GPU 텐서 메모리 할당 (PyTorch)
    dummy_input = torch.randn(tuple(input_shape), dtype=torch.float32, device='cuda')
    
    outputs = []
    for name in output_names:
        shape = engine.get_tensor_shape(name)
        outputs.append(torch.empty(tuple(shape), dtype=torch.float32, device='cuda'))
        
    # TensorRT 컨텍스트에 텐서 메모리 주소(Zero-copy 포인터) 바인딩
    context.set_tensor_address(input_name, dummy_input.data_ptr())
    for name, out_tensor in zip(output_names, outputs):
        context.set_tensor_address(name, out_tensor.data_ptr())
        
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
    print(f"[INT8] TensorRT 최종 FPS: {fps:.2f} FPS")

if __name__ == '__main__':
    onnx_path = "checkpoints/best_model_int8_fixed.onnx"
    engine_path = "checkpoints/best_model_int8_fixed.engine"
    
    # 이전에 빌드된 엔진이 있다면 삭제 후 새로 빌드
    if os.path.exists(engine_path):
        os.remove(engine_path)
        
    engine = build_engine(onnx_path, engine_path)
    if engine is not None:
        benchmark_engine(engine)
