import cv2
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import torch
import math
import numpy as np
import plotly.express as px
import datetime
from IPython.core.display import HTML
import matplotlib.pyplot as plt


# 从视频中提取帧，并跳过指定数量的帧。
def extract_frames(video_path, N):
    video_frames = []
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    current_frame = 0

    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            video_frames.append(Image.fromarray(frame[:, :, ::-1]))
        else:
            break
        current_frame += N
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    capture.release()
    return video_frames, fps


# 使用CLIP模型对帧进行编码。
def encode_frames(video_frames, model, preprocess, device, batch_size=256):
    batches = math.ceil(len(video_frames) / batch_size)
    video_features = torch.empty([0, 512], dtype=torch.float16).to(device)

    for i in range(batches):
        batch_frames = video_frames[i * batch_size: (i + 1) * batch_size]
        batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(batch_preprocessed)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
        video_features = torch.cat((video_features, batch_features))

    return video_features


# 根据搜索查询在视频帧中查找匹配项，并显示结果。
def search_video(search_query, video_features, video_frames, fps, video_url, model, device, display_heatmap=True, display_results_count=3):
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize(search_query).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = (100.0 * video_features @ text_features.T)
    values, best_photo_idx = similarities.topk(display_results_count, dim=0)

    if display_heatmap:
        print("Search query heatmap over the frames of the video:")
        fig = px.imshow(similarities.T.cpu().numpy(), height=50, aspect='auto', color_continuous_scale='viridis')
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()
        print()

    for frame_id in best_photo_idx:
        # Display the frame using matplotlib
        plt.imshow(video_frames[frame_id])
        plt.show()

        # Find the timestamp in the video and display it
        seconds = round(frame_id.cpu().numpy()[0] * N / fps)
        print(f"Found at {str(datetime.timedelta(seconds=seconds))} (link: {video_url}&t={seconds})")


if __name__ == "__main__":
    # Main execution
    video_url = "https://www.youtube.com/watch?v=PGMu_Z89Ao8"
    N = 30
    video_path = 'video.mp4'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device)
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')

    video_frames, fps = extract_frames(video_path, N)
    video_features = encode_frames(video_frames, model, preprocess, device)

    # search_video("a fire truck", video_features, video_frames, fps, video_url, model, device)
    # search_video("road works", video_features, video_frames, fps, video_url, model, device)
    # search_video("people crossing the street", video_features, video_frames, fps, video_url, model, device)
    # search_video("the Embarcadero", video_features, video_frames, fps, video_url, model, device)
    search_video("在红灯处等待", video_features, video_frames, fps, video_url, model, device, display_results_count=3)
    # search_video("a street with tram tracks", video_features, video_frames, fps, video_url, model, device)
    # search_video("the Transamerica Pyramid", video_features, video_frames, fps, video_url, model, device)