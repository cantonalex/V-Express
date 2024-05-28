import gradio as gr
import subprocess
import re

def run_inference(reference_image_path, audio_path, kps_path, retarget_strategy):
    command = [
        "python", "inference.py",
        "--reference_image_path", reference_image_path,
        "--audio_path", audio_path,
        "--kps_path", kps_path,
        "--retarget_strategy", retarget_strategy
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    
    print("Captured stdout:")
    print(result.stdout)
    print("Captured stderr:")
    print(result.stderr)
    
    output_path_match = re.search(r"--output_path\s+(\S+)", result.stdout)
    if output_path_match:
        output_path = output_path_match.group(1)
    else:
        output_path = None
    
    return result.stdout, output_path

iface = gr.Interface(
    fn=run_inference,
    inputs=[
        gr.inputs.Textbox(label="Reference Image Path", default="./test_samples/short_case/10/ref.jpg"),
        gr.inputs.Textbox(label="Audio Path", default="./test_samples/short_case/10/aud.mp3"),
        gr.inputs.Textbox(label="KPS Path", default="./test_samples/short_case/10/kps.pth"),
        gr.inputs.Dropdown(choices=["fix_face", "no_retarget", "offset_retarget", "naive_retarget"], label="Retarget Strategy", default="no_retarget")
    ],
    outputs=[
        gr.outputs.Textbox(label="Output Log"),
        gr.outputs.Video(label="Generated Video")
    ],
    title="V-Express Inference",
    description="Generate video using V-Express pipeline."
)

if __name__ == "__main__":
    iface.launch()
