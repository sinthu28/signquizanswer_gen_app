import json
import os
import cv2
import pandas as pd


META_DATA = 'data/WLASL_v0.3.json'
MISSING_INFO = 'data/missing.txt'
VIDEOS_FOLDER = 'data/videos/'


def load_metadata(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_missing_info(path):
    with open(path, 'r') as f:
        return set(line.strip() for line in f)


def handle_missing_info(videoIDs, missingIDs):
    result = []
    for videoID in videoIDs:
        if videoID not in missingIDs:
            result.append(videoID)
    return result


def create_dataset(meta_path, missingID_path):
    result = []
    meta_data = load_metadata(path=meta_path)
    missing_data = load_missing_info(path=missingID_path)
    for entry in meta_data:
        label = entry['gloss']
        for instance in entry['instances']:
            bbox = instance['bbox']
            videoID = instance['video_id']
            if videoID not in missing_data:
                result.append({
                    'video_id': videoID,
                    'label': label,
                    'bbox': bbox
                })
    
    return result


def create_video_features(video_folder_path, meta_path, missingID_path):
    result = []
    cleaned_data = create_dataset(meta_path, missingID_path)
    for data in cleaned_data:
        label = data['label']
        bbox = data['bbox']
        video_id = data['video_id']
        video_path = os.path.join(video_folder_path, video_id + '.mp4')

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        result.append({
            'label': label,
            'video_id': video_id,
            'bbox': bbox,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count
        })
    
    return result


def create_video_metadata(metadata_path, video_path, missing_txt):
    results = []

    metadata = load_metadata(metadata_path)
    data = create_video_features(video_path, metadata_path, missing_txt)

    for entry in metadata:
        for instance in entry['instances']:
            video_id = instance['video_id']
            frame_start = instance['frame_start']
            frame_end = instance['frame_end']

            for record in data:
                if video_id in record['video_id']:
                    results.append({
                        'video_id': video_id,
                        'label': record['label'],
                        'bbox': record['bbox'],
                        'width': record['width'],
                        'height': record['height'],
                        'fps': record['fps'],
                        'frame_count': record['frame_count'],
                        'frame_start': frame_start,
                        'frame_end': frame_end
                    })
    
    return results
                    

meta = create_video_metadata(metadata_path=META_DATA, video_path=VIDEOS_FOLDER, missing_txt=MISSING_INFO)

with open('MetaData.json', 'w') as json_file:
    json.dump(meta, json_file, indent=4)

print("Data saved as MetaData.json")

