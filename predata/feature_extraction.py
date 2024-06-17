import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision import transforms

def load_model():
    """
    加载预训练的面部表情分类模型。
    
    返回:
        tuple: 包含处理器和模型的元组。
    """
    processor = AutoImageProcessor.from_pretrained("/Volumes/SD扩展/pre-train/vit-face-expression")
    model = AutoModelForImageClassification.from_pretrained("/Volumes/SD扩展/pre-train/vit-face-expression")
    return processor, model

# 定义图像预处理的转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_frames(frames):
    """
    预处理视频帧。
    
    参数:
        frames (list): 包含视频帧的列表。
    
    返回:
        list: 预处理后的帧列表。
    """
    return [transform(frame) for frame in frames]

def deprocess_tensor(tensor):
    """
    对张量进行反归一化处理。
    
    参数:
        tensor (Tensor): 输入的归一化张量。
    
    返回:
        Tensor: 反归一化后的张量。
    """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    return tensor

def extract_features(frames, processor, model):
    """
    从预处理后的帧中提取特征。
    
    参数:
        frames (list): 预处理后的帧列表。
        processor (AutoImageProcessor): 预训练模型的处理器。
        model (AutoModelForImageClassification): 预训练模型。
    
    返回:
        Tensor: 提取的特征张量。
    """
    deprocessed_frames = [deprocess_tensor(frame) for frame in frames]
    inputs = processor(images=deprocessed_frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]  # 提取最后一层的隐藏状态作为特征
    return hidden_states
