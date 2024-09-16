import json
from typing import List, Dict, Any
from collections.abc import Set
import os
from Loader.VideoLoader import VideoLoader
import concurrent.futures

class WLASLDatasetLoader:
    def __init__(self, json_path: str, missing_file_path: str, video_dir: str, log_dir = 'logs', max_workers: int = 4, batch_size: int = 10):
        self.json_path = json_path
        self.missing_file_path = missing_file_path
        self.video_dir = video_dir
        self.metadata = self._load_json()
        self.missing_videos = self._load_missing_videos()
        self.video_loader = VideoLoader(log_dir=log_dir)
        self.max_workers = max_workers
        self.batch_size = batch_size

    def _load_json(self) -> List[Dict[str, Any]]:
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def _load_missing_videos(self) -> Set[str]:
        with open(self.missing_file_path, 'r') as f:
            return set(line.strip() for line in f)

    def _get_video_path(self, video_id: str) -> str:
        return os.path.join(self.video_dir, f"{video_id}.mp4")

    def _load_frames_for_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        video_id = instance['video_id']
        if video_id in self.missing_videos:
            return None

        video_path = self._get_video_path(video_id)
        frames = self.video_loader.load_frames(video_path)
        
        if frames is not None:
            return {
                'gloss': instance.get('gloss', 'unknown'),
                'instances': instance,
                'frames': frames
            }
        else:
            print(f"Warning: Unable to load frames for video {video_id}")
            return None

    def load_dataset(self, limit=None) -> List[Dict[str, Any]]:
        dataset = []
        batch = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for entry in self.metadata:
                for instance in entry['instances']:
                    if limit and len(dataset) >= limit:
                        return dataset
                    
                    batch.append(instance)
                    if len(batch) >= self.batch_size:
                        futures = [executor.submit(self._load_frames_for_instance, inst) for inst in batch]
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                result = future.result()
                                if result is not None:
                                    dataset.append(result)
                            except Exception as e:
                                self.logger.error(f"Error loading frames for instance: {e}")
                        batch = []
            
            if batch:
                futures = [executor.submit(self._load_frames_for_instance, inst) for inst in batch]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            dataset.append(result)
                    except Exception as e:
                        self.logger.error(f"Error loading frames for instance: {e}")
        
        return dataset

    def get_statistics(self) -> Dict[str, int]:
        total_videos = sum(len(entry['instances']) for entry in self.metadata)
        loaded_videos = len([instance for entry in self.metadata for instance in entry['instances'] if instance['video_id'] not in self.missing_videos])
        missing_videos = len(self.missing_videos)
        
        return {
            'total_videos': total_videos,
            'loaded_videos': loaded_videos,
            'missing_videos': missing_videos
        }


#     def load_metadata_json(self):   
#         json_path = '/Users/dxt/Desktop/beta_/data/WLASL_v0.3.json'
#         with open(json_path, 'r') as metadata:
#             return json.load(metadata)
    
#     def prepare_dataset(self):
#         for entry in self.load_metadata_json:
#             for instance in entry['instances']:
#                 print(type(instance))
#         # pass



# test = WLASLDatasetLoader()
# test.prepare_dataset()