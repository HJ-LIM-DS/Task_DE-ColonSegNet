import onnx
import numpy as np
from onnx import numpy_helper

def fix_bias_qdq(input_path, output_path):
    print("TensorRT 파싱 오류 수정을 위해 Bias의 QDQ 노드를 원래의 FP32 값으로 복원합니다...")
    model = onnx.load(input_path)
    graph = model.graph
    
    # 이니셜라이저 딕셔너리 구성
    init_dict = {init.name: init for init in graph.initializer}
    
    # Conv 레이어의 Bias(3번째 인풋)로 들어가는 텐서 이름 수집
    conv_biases = set()
    for node in graph.node:
        if node.op_type in ["Conv", "ConvTranspose", "Gemm"] and len(node.input) >= 3:
            conv_biases.add(node.input[2])
            
    nodes_to_remove = []
    new_initializers = []
    
    for node in graph.node:
        if node.op_type == "DequantizeLinear" and node.output[0] in conv_biases:
            bias_name = node.output[0]
            q_name = node.input[0]
            scale_name = node.input[1]
            zp_name = node.input[2] if len(node.input) > 2 else None
            
            if q_name in init_dict and scale_name in init_dict:
                q_tensor = numpy_helper.to_array(init_dict[q_name])
                scale_tensor = numpy_helper.to_array(init_dict[scale_name])
                
                if zp_name and zp_name in init_dict:
                    zp_tensor = numpy_helper.to_array(init_dict[zp_name])
                else:
                    zp_tensor = np.zeros_like(q_tensor)
                    
                # Dequantize 공식: (x - zero_point) * scale
                fp32_tensor = (q_tensor.astype(np.float32) - zp_tensor.astype(np.float32)) * scale_tensor.astype(np.float32)
                
                # 새로운 FP32 이니셜라이저 생성 (원래 Bias 이름으로)
                new_init = numpy_helper.from_array(fp32_tensor, name=bias_name)
                new_initializers.append(new_init)
                nodes_to_remove.append(node)
                print(f"[수정됨] {bias_name} (Int32 -> FP32 복원 완료)")
                
    # 노드 제거
    for node in nodes_to_remove:
        graph.node.remove(node)
        
    # 새 이니셜라이저 추가
    for new_init in new_initializers:
        graph.initializer.append(new_init)
        
    onnx.save(model, output_path)
    print(f"수정된 ONNX 모델이 저장되었습니다: {output_path}")

if __name__ == "__main__":
    in_path = "checkpoints/best_model_int8.onnx"
    out_path = "checkpoints/best_model_int8_fixed.onnx"
    fix_bias_qdq(in_path, out_path)
