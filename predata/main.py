import os
import pickle
from video_processing import extract_frames
from feature_extraction import preprocess_frames, extract_features, load_model
from tqdm import tqdm
def get_video_info(root_dir):
    """
    获取所有视频文件的信息。
    
    参数:
        root_dir (str): 包含所有视频文件夹的根目录。
    
    返回:
        list: 包含每个视频文件信息的字典列表。
    """
    video_info = []
    for video_folder in os.listdir(root_dir):
        video_folder_path = os.path.join(root_dir, video_folder)
        if os.path.isdir(video_folder_path):
            video_id = video_folder
            for video_file in os.listdir(video_folder_path):
                if video_file.endswith(".mp4"):
                    clip_id = os.path.splitext(video_file)[0]
                    video_info.append({"video_id": video_id, "clip_id": clip_id, "file_path": os.path.join(video_folder_path, video_file)})
    return video_info

def process_and_collect_features(video_info, processor, model):
    """
    处理视频并收集特征。
    
    参数:
        video_info (list): 包含视频文件信息的字典列表。
        processor (AutoImageProcessor): 预训练模型的处理器。
        model (AutoModelForImageClassification): 预训练模型。
    
    返回:
        dict: 包含每个视频片段特征的字典。
    """
    all_features = {}
    
    for info in tqdm(video_info,desc='Processing videos'):
        video_path = info["file_path"]
        video_id = info["video_id"]
        clip_id = info["clip_id"]
        print(video_path)
        # 提取视频帧
        frames = extract_frames(video_path)
        
        # 预处理帧
        preprocessed_frames = preprocess_frames(frames)
        
        # 提取特征
        features = extract_features(preprocessed_frames, processor, model)

        # 将特征存储在字典中
        all_features[f"{video_id}_{clip_id}"] = features

        print(f"Processed features for {video_id}/{clip_id}")

    return all_features

def main():
    """
    主函数，执行整个视频处理和特征提取流程。
    """
    root_directory = '/Volumes/SD扩展/datasets/MSADatasets/MOSI/Raw'
    output_file = 'video_features.pkl'
    
    # 获取视频信息
    video_info_list = get_video_info(root_directory)
    
    # 加载模型和处理器
    processor, model = load_model()
    
    # 处理视频并收集特征
    all_features = process_and_collect_features(video_info_list, processor, model)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存所有特征到一个pkl文件
    with open(output_file, 'wb') as f:
        pickle.dump(all_features, f)
    print(f"Saved all features to {output_file}")

if __name__ == "__main__":
    main()
