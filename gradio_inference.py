import gradio as gr
import subprocess
import re

def run_inference(reference_image_path, audio_path, kps_path, retarget_strategy):
    command = [
        "python", "inference.py",
        "--reference_image_path", reference_image_path,
        "--audio_path", audio_path,
        "--kps_path", kps_path,
        "--retarget_strategy", retarget_strategy,
        "--output_path", "output.mp4"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    
    print("Captured stdout:")
    print(result.stdout)
    print("Captured stderr:")
    print(result.stderr)
    
    output_path = "output.mp4"
    
    return result.stdout, output_path

iface = gr.Interface(
    fn=run_inference,
    inputs=[
        gr.Textbox(label="Reference Image Path", value="./test_samples/short_case/10/ref.jpg"),
        gr.Textbox(label="Audio Path", value="./test_samples/short_case/10/aud.mp3"),
        gr.Textbox(label="KPS Path", value="./test_samples/short_case/10/kps.pth"),
        gr.Dropdown(choices=["fix_face", "no_retarget", "offset_retarget", "naive_retarget"], label="Retarget Strategy", value="no_retarget")
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
