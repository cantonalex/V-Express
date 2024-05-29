import argparse
import os
import cv2
import torch
from insightface.app import FaceAnalysis
from imageio_ffmpeg import get_ffmpeg_exe
import gradio as gr
import subprocess

def extract_kps_and_audio(video_path, kps_sequence_save_path, audio_save_path, device='cuda', gpu_id=0, insightface_model_path='./model_ckpts/insightface_models/', height=512, width=512):
    # Run the extraction script with default paths from the documentation
    command = [
        "python", "scripts/extract_kps_sequence_and_audio.py",
        "--video_path", video_path,
        "--kps_sequence_save_path", kps_sequence_save_path,
        "--audio_save_path", audio_save_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    output_log = f"Captured stdout:\n{result.stdout}\n\nCaptured stderr:\n{result.stderr}"
    return output_log, kps_sequence_save_path, audio_save_path

def run_inference(reference_image_path, audio_path, kps_path):
    # Run the inference script with necessary parameters
    command = [
        "python", "inference.py",
        "--reference_image_path", reference_image_path,
        "--audio_path", audio_path,
        "--kps_path", kps_path,
        "--output_path", "output.mp4",
        "--num_inference_steps", "10",  
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    output_log = f"Captured stdout:\n{result.stdout}\n\nCaptured stderr:\n{result.stderr}"
    output_path = "output.mp4"
    return output_log, output_path

def process_video(video_path, reference_image_path):
    # Use default paths for the keypoint and audio save locations
    kps_path = "./test_samples/short_case/10/kps.pth"
    audio_path = "./test_samples/short_case/10/aud.mp3"
    
    extract_log, kps_path, audio_path = extract_kps_and_audio(video_path, kps_path, audio_path)
    inference_log, output_path = run_inference(reference_image_path, audio_path, kps_path)
    
    output_log = f"{extract_log}\n\n{inference_log}"
    return output_log, output_path

iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Textbox(label="Video Path", value="./test_samples/short_case/10/gt.mp4"),
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
