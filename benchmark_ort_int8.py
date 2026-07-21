import os
import glob
import cv2
import numpy as np
import time
import torch
import onnx
import onnxruntime as ort
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

def benchmark_ort(model_path, precision_name):
    print(f"[{precision_name}] ONNX Runtime 추론 엔진 로드 중...")
    
    providers = ['CUDAExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
    dummy_input = np.random.randn(1, 3, 352, 352).astype(np.float32)
    
    print(f"[{precision_name}] 웜업(Warm-up) 중...")
    for _ in range(50):
        session.run(None, {input_name: dummy_input})
        
    print(f"[{precision_name}] 본격적인 성능(FPS) 벤치마킹 시작 (1000 프레임)...")
    num_iters = 1000
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iters):
        session.run(None, {input_name: dummy_input})
    torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_iters / total_time
    ms_per_frame = (total_time / num_iters) * 1000
    
    print("\n" + "=" * 50)
    print(f"[{precision_name}] 벤치마크 결과:")
    print(f"총 {num_iters} 프레임 처리 시간: {total_time:.2f} 초")
    print(f"단일 프레임 평균 처리 시간: {ms_per_frame:.2f} ms")
    print(f"달성 FPS: {fps:.2f} FPS")
    print("=" * 50)
    
    return fps

if __name__ == '__main__':
    onnx_path = "checkpoints/best_model.onnx"
    int8_onnx_path = "checkpoints/best_model_int8.onnx"
    data_dir = "E:/Task_DE-ColonSegNet/datasets/SUN-SEG_dataset"
    
    if not os.path.exists(int8_onnx_path):
        quantize_onnx_to_int8(onnx_path, int8_onnx_path, data_dir)
        
    print("=== 원본 FP32 모델 속도 측정 ===")
    fps_fp32 = benchmark_ort(onnx_path, "FP32")
    
    print("\n=== 양자화된 INT8 모델 속도 측정 ===")
    fps_int8 = benchmark_ort(int8_onnx_path, "INT8")
    
    print("\n[최종 성능 비교]")
    print(f"FP32 FPS: {fps_fp32:.2f}  ->  INT8 FPS: {fps_int8:.2f}")
    print(f"성능 향상: {fps_int8 / fps_fp32:.2f} 배 빨라졌습니다.")
