from collections.abc import Set
import json
import os
from typing import List, Dict, Any
from VideoLoader import VideoLoader

class WLASLDatasetLoader:
    def __init__(self, json_path: str, missing_file_path: str, video_dir: str):
        self.json_path = json_path
        self.missing_file_path = missing_file_path
        self.video_dir = video_dir
        self.metadata = self._load_json()
        self.missing_videos = self._load_missing_videos()
        self.video_loader = VideoLoader()

    def _load_json(self) -> List[Dict[str, Any]]:
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def _load_missing_videos(self) -> Set[str]:
        with open(self.missing_file_path, 'r') as f:
            return set(line.strip() for line in f)

    def _get_video_path(self, video_id: str) -> str:
        return os.path.join(self.video_dir, f"{video_id}.mp4")

    def load_dataset(self) -> List[Dict[str, Any]]:
        dataset = []
        for entry in self.metadata:
            instances = entry['instances']
            for instance in instances:
                video_id = instance['video_id']
                if video_id in self.missing_videos:
                    continue
                
                video_path = self._get_video_path(video_id)
                frames = self.video_loader.load_frames(video_path)
                
                if frames is not None:
                    dataset_entry = {
                        'gloss': entry['gloss'],
                        'instances': instance,
                        'frames': frames
                    }
                    dataset.append(dataset_entry)
                else:
                    print(f"Warning: Unable to load frames for video {video_id}")
        
        return dataset

    def get_statistics(self) -> Dict[str, int]:
        total_videos = sum(len(entry['instances']) for entry in self.metadata)
        loaded_videos = len([entry for entry in self.metadata 
                             for instance in entry['instances'] 
                             if instance['video_id'] not in self.missing_videos])
        missing_videos = len(self.missing_videos)
        
        return {
            'total_videos': total_videos,
            'loaded_videos': loaded_videos,
            'missing_videos': missing_videos
        }

# ###########################################################################################
# #                                   #                                                    #
# ###########################################################################################

# # Usage example:
# metaData = '/Users/dxt/Desktop/beta_/data/WLASL_v0.3.json'
# missingData = '/Users/dxt/Desktop/beta_/data/missing.txt'
# videoDir = '/Users/dxt/Desktop/beta_/data/videos'

# wlasl_loader = WLASLDatasetLoader(metaData, missingData, videoDir)
# dataset = wlasl_loader.load_dataset()
# stats = wlasl_loader.get_statistics()
# print(f"Dataset statistics: {stats}")