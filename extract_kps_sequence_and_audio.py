import argparse
import os
import cv2
import torch
from insightface.app import FaceAnalysis
from imageio_ffmpeg import get_ffmpeg_exe
import gradio as gr
import subprocess

def extract_kps_and_audio(video_path, kps_sequence_save_path, audio_save_path, device='cuda', gpu_id=0, insightface_model_path='./model_ckpts/insightface_models/', height=512, width=512):
    app = FaceAnalysis(
        providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'],
        provider_options=[{'device_id': gpu_id}] if device == 'cuda' else [],
        root=insightface_model_path,
    )
    app.prepare(ctx_id=0, det_size=(height, width))
    os.system(f'{get_ffmpeg_exe()} -i "{video_path}" -y -vn "{audio_save_path}"')
    
    kps_sequence = []
    video_capture = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        faces = app.get(frame)
        assert len(faces) == 1, f'There are {len(faces)} faces in the {frame_idx}-th frame. Only one face is supported.'
        kps = faces[0].kps[:3]
        kps_sequence.append(kps)
        frame_idx += 1
    
    torch.save(kps_sequence, kps_sequence_save_path)
    
    return kps_sequence_save_path, audio_save_path

def run_inference(video_path, kps_sequence_save_path, audio_save_path):
    command = [
        "python", "inference.py",
        "--video_path", video_path,
        "--kps_sequence_save_path", kps_sequence_save_path,
        "--audio_save_path", audio_save_path,
        "--output_path", "output.mp4"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    output_log = f"Captured stdout:\n{result.stdout}\n\nCaptured stderr:\n{result.stderr}"
    output_path = "output.mp4"
    return output_log, output_path

def process_video(video_path, reference_image_path):
    kps_path, audio_path = extract_kps_and_audio(video_path, "kps.pth", "audio.mp3")
    return run_inference(video_path, kps_path, audio_path)

iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Textbox(label="Video Path", value="./test_samples/short_case/10/video.mp4"),
        gr.Textbox(label="Reference Image Path", value="./test_samples/short_case/10/ref.jpg")
    ],
    outputs=[
        gr.Textbox(label="Output Log"),
        gr.Video(label="Generated Video")
    ],
    title="V-Express Inference",
    description="Generate video using V-Express pipeline."
)

if __name__ == "__main__":
    iface.launch()
