#!/usr/bin/env python3
"""
Video Processor - Run with same command as main.py
Usage: sudo uv run --with-requirements requirements.txt --python 3.12 process_video.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe <video_path>
"""

import torch
import hydra
import os
import sys
from datetime import datetime
from ollama import chat
from pydantic import BaseModel
from pipelines.pipeline import InferencePipeline


class Lip2TextOutput(BaseModel):
    list_of_changes: str
    corrected_text: str


class VideoProcessor:
    def __init__(self, vsr_model):
        self.vsr_model = vsr_model
    
    def process_video(self, video_path):
        """
        Process a single video file using the actual VSR model
        """
        print(f"🎬 Processing video: {video_path}")
        
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print("🔍 Running lip-to-text inference with VSR model...")
        
        # Use the actual VSR model to process the video
        output = self.vsr_model(video_path)
        
        if not output or output.strip() == "":
            print("❌ No speech detected in the video")
            return {"output": "No speech detected", "video_path": video_path}
        
        print(f"📝 Raw transcription: {output}")
        
        # Use the same LLM correction as main.py
        print("🧠 Correcting transcription with LLM...")
        
        try:
            response = chat(
                model='llama3.2',
                messages=[
                    {
                        'role': 'system',
                        'content': f"You are an assistant that helps make corrections to the output of a lipreading model. The text you will receive was transcribed using a video-to-text system that attempts to lipread the subject speaking in the video, so the text will likely be imperfect.\n\nYour task is to:\n1. Fix any obvious transcription errors\n2. Add proper punctuation\n3. Capitalize correctly (not all caps)\n4. Make the text more readable\n\nDo NOT change the meaning or add new information. Only correct obvious errors and improve readability.\n\nReturn the corrected text in the format of 'list_of_changes' and 'corrected_text'."
                    },
                    {
                        'role': 'user',
                        'content': f"Please correct this lipreading transcription:\n\n{output}"
                    }
                ],
                format=Lip2TextOutput.model_json_schema()
            )
            
            # Get only the corrected text
            chat_output = Lip2TextOutput.model_validate_json(
                response.message.content)
            
            # If last character isn't a sentence ending, add a period
            if chat_output.corrected_text and chat_output.corrected_text[-1] not in ['.', '?', '!']:
                chat_output.corrected_text += '.'
                
        except Exception as e:
            print(f"⚠️ LLM correction failed: {e}")
            print("Using raw transcription instead...")
            chat_output = Lip2TextOutput(
                list_of_changes="LLM correction failed",
                corrected_text=output
            )
        
        print(f"✨ Corrected transcription: {chat_output.corrected_text}")
        
        return {
            "output": chat_output.corrected_text,
            "video_path": video_path
        }


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    print("🎥 Lip-to-Text Video Processor")
    print("=" * 60)
    
    # Load the VSR model (same as main.py)
    print("🔄 Loading VSR model...")
    vsr_model = InferencePipeline(
        cfg.config_filename, 
        device=torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() and cfg.gpu_idx >= 0 else "cpu"), 
        detector=cfg.detector, 
        face_track=True
    )
    print("✅ VSR model loaded successfully!")
    
    # Initialize video processor
    processor = VideoProcessor(vsr_model)
    
    # Get video path from hydra overrides
    video_path = None
    
    # Check if video path was provided as a hydra override
    if hasattr(cfg, 'video_path') and cfg.video_path:
        video_path = cfg.video_path
    else:
        # Try to find video path in command line arguments
        for arg in sys.argv[1:]:
            if not arg.startswith('config_filename=') and not arg.startswith('detector=') and not arg.startswith('gpu_idx=') and not arg.startswith('video_path=') and not arg.startswith('+video_path='):
                if os.path.exists(arg) or arg.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = arg
                    break
    
    if not video_path:
        print("❌ Please provide a video file path as an argument:")
        print("   sudo uv run --with-requirements requirements.txt --python 3.12 process_video.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe +video_path=path/to/video.mp4")
        return
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file '{video_path}' not found!")
        return
    
    print(f"🎬 Processing video: {video_path}")
    print("=" * 60)
    
    try:
        # Process the video
        result = processor.process_video(video_path)
        
        print("=" * 60)
        print("📄 FINAL TRANSCRIPTION:")
        print("=" * 60)
        print(result["output"])
        print("=" * 60)
        
        # Save the result to a text file
        output_filename = f"transcription_{os.path.splitext(os.path.basename(video_path))[0]}.txt"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"Video: {video_path}\n")
            f.write(f"Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Transcription:\n{result['output']}\n")
        
        print(f"💾 Transcription saved to: {output_filename}")
        
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        return


if __name__ == '__main__':
    main()
