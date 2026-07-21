import os
import torch
from models.de_colonsegnet import DE_ColonSegNet

def export_to_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장비: {device}")

    # 모델 생성 및 가중치 로드
    model = DE_ColonSegNet(in_channels=3, out_channels=1).to(device)
    checkpoint_path = "checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"체크포인트 {checkpoint_path} 가 없습니다.")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("모델 가중치 로드 완료.")

    # 더미 입력 텐서 생성 (배치사이즈 1, 채널 3, 352x352)
    dummy_input = torch.randn(1, 3, 352, 352, device=device)
    onnx_path = "checkpoints/best_model.onnx"

    print(f"ONNX 변환을 시작합니다. 대상 파일: {onnx_path}")
    
    # torch.onnx.export 실행
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=14,          # ONNX opset 버전 (최신 TensorRT 호환을 위해 11 이상 권장)
        do_constant_folding=True,  # 상수 폴딩 최적화
        input_names=['input'],     # 입력 노드 이름
        output_names=['output', 'aux_output']   # 출력 노드 이름 (DE_ColonSegNet은 두 개 출력: main, aux)
    )
    
    print("ONNX 변환 완료!")

if __name__ == "__main__":
    export_to_onnx()
