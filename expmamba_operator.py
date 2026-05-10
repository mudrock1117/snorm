import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# 添加ExpoMamba路径
sys.path.append('/media/zdh/f37daf80-9fee-4790-aa5a-bcedd13aa82e/2026/hdx/ExpoMamba/ExpoMamba')

# 直接导入ExpoMamba
from models.expomamba_2d import ExpoMamba
print("✅ 成功导入ExpoMamba")

class ExpoMambaOperator:
    def __init__(self, checkpoint_path=None, device='cuda', base_channels=48, n_channels=3):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ExpoMamba(
            n_channels=n_channels, 
            bilinear=True, 
            base_channels=base_channels
        ).to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        else:
            print(f"⚠️  未找到权重文件: {checkpoint_path}")
            print("使用随机初始化权重")
        
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # 处理不同格式的权重
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ 成功加载权重: {checkpoint_path}")
        except Exception as e:
            print(f"❌ 加载权重失败: {e}")
            print("使用随机初始化权重")

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            image = self.transform(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        return image

    def postprocess(self, tensor):
        tensor = tensor.squeeze(0).cpu()
        tensor = torch.clamp(tensor, 0, 1)
        return transforms.ToPILImage()(tensor)

    @torch.no_grad()
    def enhance(self, image, return_tensor=False):
        input_tensor = self.preprocess(image)
        output_tensor = self.model(input_tensor)
        if return_tensor:
            return output_tensor
        else:
            return self.postprocess(output_tensor)

    def batch_enhance(self, images, batch_size=8):
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensors = [self.preprocess(img) for img in batch]
            batch_input = torch.cat(batch_tensors, dim=0)
            
            with torch.no_grad():
                batch_output = self.model(batch_input)
            
            for j in range(batch_output.size(0)):
                results.append(self.postprocess(batch_output[j:j+1]))
        
        return results

    def enhance_tensor(self, tensor):
        """直接处理张量，用于集成到现有模型中"""
        with torch.no_grad():
            tensor = tensor.to(self.device)
            output = self.model(tensor)
        return output

# 创建全局算子实例
expmamba_operator = None

def get_expmamba_operator(checkpoint_path=None):
    global expmamba_operator
    if expmamba_operator is None:
        # 使用ExpoMamba项目的预训练权重
        default_checkpoint = '/media/zdh/f37daf80-9fee-4790-aa5a-bcedd13aa82e/2026/hdx/ExpoMamba/ExpoMamba/checkpoint/lolv1/lolv1_5_Apr_18h_37m/best_ssim.pth'
        expmamba_operator = ExpoMambaOperator(checkpoint_path or default_checkpoint)
    return expmamba_operator

if __name__ == "__main__":
    # 测试算子
    operator = ExpoMambaOperator()
    test_image = Image.new('RGB', (512, 512), color='gray')
    enhanced = operator.enhance(test_image)
    enhanced.save('test_expmamba_output.jpg')
    print("✅ 测试完成，输出保存为 test_expmamba_output.jpg")
