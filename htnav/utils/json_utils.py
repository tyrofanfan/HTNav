import json
import os

class SceneDescriptorLoader:
    def __init__(self, json_path="citynav/data/newcityrefer/processed_descriptions.json"):
        # 确保路径正确
        self.json_path = json_path
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at: {json_path}")
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
    
    def get_scene_descriptor(self, scene_id, view_id="0", descriptor_index=0):
        """
        获取特定场景和视图的描述符
        """
        # 获取场景数据
        scene_data = self.data.get(scene_id)
        if not scene_data:
            available_scenes = list(self.data.keys())
            raise ValueError(f"Scene {scene_id} not found. Available scenes: {available_scenes}")
        
        # 获取视图数据
        view_data = scene_data.get(view_id)
        if not view_data:
            available_views = list(scene_data.keys())
            raise ValueError(f"View {view_id} not found for scene {scene_id}. Available views: {available_views}")
        
        # 获取描述符
        if descriptor_index >= len(view_data):
            descriptor_index = 0  # 默认使用第一个描述符
            print(f"Warning: descriptor_index {descriptor_index} out of range. Using first descriptor.")
        
        descriptor = view_data[descriptor_index]
        return {
            "target": descriptor["target"],
            "landmarks": descriptor["landmarks"],
            "surroundings": descriptor["surroundings"]
        }
    
    def get_all_scene_ids(self):
        """获取所有可用的场景ID"""
        return list(self.data.keys())
    
    def get_views_for_scene(self, scene_id):
        """获取场景的所有视图ID"""
        scene_data = self.data.get(scene_id, {})
        return list(scene_data.keys())