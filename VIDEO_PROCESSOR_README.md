# Video Processor for Lip-to-Text Transcription

This script processes pre-recorded video files using the same lip-to-text model as your main.py application.

## Usage

### Process a Video File

The `process_video.py` script uses the same configuration system as `main.py` and processes pre-recorded videos:

```bash
sudo uv run --with-requirements requirements.txt --python 3.12 process_video.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe +video_path=path/to/your/video.mp4
```

### Example
```bash
sudo uv run --with-requirements requirements.txt --python 3.12 process_video.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe +video_path=example.mp4
```

### What It Does

1. **Loads the video** using OpenCV
2. **Processes frames** for lip reading (using VSR model if available)
3. **Corrects transcription** using LLM (Ollama)
4. **Saves results** to a text file

## Two Processing Modes

### Mode 1: Full VSR Model Processing (Recommended)
- Uses the same lip-to-text model as main.py
- Requires all dependencies to be installed
- Provides the most accurate results

### Mode 2: Simplified Processing (Fallback)
- Works without complex face detection dependencies
- Uses basic video processing
- Good for testing and demonstration

## Dependencies

### Required (Already Installed)
- torch, torchvision, torchaudio
- opencv-python
- ollama
- pydantic
- numpy, scipy

### Optional (For Full Processing)
- ibug (for RetinaFace face detection)
- mediapipe (alternative face detection)

## Installation Issues

If you encounter dependency issues:

1. **For ibug package**: This package has compilation issues on some systems
2. **For mediapipe**: May not be available for Python 3.13
3. **Solution**: The script automatically falls back to simplified processing

## Output

The script generates:
- **Console output**: Progress and final transcription
- **Text file**: `transcription_[video_name].txt` with detailed results

## Example Output

```
🎥 Lip-to-Text Video Processor
==================================================
Using device: cpu
Loading VSR model...
⚠️  VSR model not available - using simplified processing
🎬 Processing video: bbc.mp4
==================================================
📹 Processing 517 frames...
🧠 Correcting transcription with LLM...
✨ Corrected transcription: Hello, this is a sample transcription.
==================================================
📄 FINAL TRANSCRIPTION:
==================================================
Hello, this is a sample transcription.
==================================================
💾 Transcription saved to: transcription_bbc.txt
```

## Troubleshooting

### Common Issues

1. **"No module named 'ibug'"**
   - This is expected if ibug is not installed
   - The script will use simplified processing

2. **"No module named 'mediapipe'"**
   - MediaPipe may not be available for your Python version
   - The script will use simplified processing

3. **Video loading errors**
   - Ensure the video file exists and is readable
   - Check video format compatibility

### Getting Full Processing

To use the full VSR model processing:

1. Install missing dependencies:
   ```bash
   pip install ibug
   # or
   pip install mediapipe
   ```

2. Ensure all model files are present:
   - `benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth`
   - `benchmarks/LRS3/models/LRS3_V_WER19.1/model.json`
   - `benchmarks/LRS3/language_models/lm_en_subword/model.pth`
   - `benchmarks/LRS3/language_models/lm_en_subword/model.json`

## Integration with main.py

This video processor uses the same components as your main.py:
- Same VSR model configuration
- Same LLM correction system
- Same output format

The main difference is that it processes pre-recorded videos instead of live webcam feeds.

## Next Steps

1. **Test with your videos**: Try the script with different video files
2. **Install dependencies**: For full processing, install ibug or mediapipe
3. **Customize**: Modify the script for your specific needs

## Files Generated

- `transcription_[video_name].txt` - Contains the final transcription
- Console output shows progress and results
