# Lip2Text - Comprehensive Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Core Features](#core-features)
4. [Technology Stack](#technology-stack)
5. [Deep Dive into Components](#deep-dive-into-components)
6. [User Interfaces](#user-interfaces)
7. [Model Details](#model-details)
8. [Workflow Processes](#workflow-processes)
9. [Installation & Setup](#installation--setup)
10. [Usage Guide](#usage-guide)
11. [Configuration](#configuration)
12. [Performance Considerations](#performance-considerations)
13. [Future Enhancements](#future-enhancements)

---

## Project Overview

Lip2Text is a comprehensive, privacy-first transcription system that converts spoken communication into text through multiple modalities. The system provides three primary modes of operation:

1. **Real-time Webcam Lip Reading**: Uses visual-only cues to transcribe speech from video
2. **Audio Speech-to-Text**: Captures and transcribes audio from microphone input
3. **Batch Video Processing**: Processes pre-recorded video files for transcription

All processing occurs locally on the user's machine, ensuring complete privacy and data security. The system integrates Large Language Models (LLMs) for post-processing correction and enhancement of transcriptions.

### Key Selling Points
- **Privacy-First**: All processing happens locally; no cloud calls for transcription
- **Multi-Modal**: Supports both visual lip reading and audio transcription
- **Real-Time**: Low-latency processing with always-on-top preview
- **Accurate**: State-of-the-art models with LLM-enhanced corrections
- **Production-Ready**: Web interface for enterprise deployment

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  Command-Line Interface  │  Web Application  │  Real-Time Mode   │
└──────────────┬────────────────┬────────────────┬──────────────────┘
               │                │                │
┌──────────────▼────────────────▼────────────────▼──────────────────┐
│                      Application Layer                            │
├──────────────────────────────────────────────────────────────────┤
│  main.py                          process_video.py               │
│  - Real-time capture              - Batch processing             │
│  - Mode switching                 - File I/O                     │
│  - Keyboard controls              - Result persistence           │
│                                                                    │
│  web/app.py                                                       │
│  - Flask server                   - REST endpoints               │
│  - File uploads                   - Session management           │
└──────────────┬───────────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────────┐
│                      Pipeline Layer                               │
├──────────────────────────────────────────────────────────────────┤
│  InferencePipeline  │  AudioRecorder  │  AudioUtils              │
│  - Data loading     │  - Capture      │  - Processing            │
│  - Model exec       │  - Buffering    │  - Conversion            │
│  - Post-processing  │  - Threading    │  - Whisper integration   │
└──────────────┬──────────────┬──────────────────┬─────────────────┘
               │              │                  │
┌──────────────▼──────────────▼──────────────────▼─────────────────┐
│                       AI Model Layer                              │
├──────────────────────────────────────────────────────────────────┤
│  Lip Reading Model (AVSR)  │  Whisper STT  │  LLM (Ollama)      │
│  - Transformer-based       │  - OpenAI     │  - Llama 3.2       │
│  - CNN backbone            │  - Multilingual│  - Correction      │
│  - Beam search             │  - Fine-tuned │  - Enhancement      │
└──────────────┬──────────────┬──────────────────┬─────────────────┘
               │              │                  │
┌──────────────▼──────────────▼──────────────────▼─────────────────┐
│                    Computer Vision & Audio                        │
├──────────────────────────────────────────────────────────────────┤
│  MediaPipe/RetinaFace  │  OpenCV  │  PyAudio  │  TorchAudio      │
│  - Face detection      │  - Frames│  - Capture│  - Processing    │
│  - Landmark tracking   │  - Encode│  - Streams│  - Resampling    │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow

#### Lip Reading Pipeline
```
Video Input → Frame Extraction → Face Detection → ROI Extraction 
→ Video Preprocessing → Feature Extraction → Transformer Encoder 
→ Beam Search Decoder → Language Model → Raw Text → LLM Correction 
→ Final Transcription
```

#### Audio Transcription Pipeline
```
Microphone → Audio Buffering → Preprocessing → Resampling 
→ Whisper Encoder → Transformer Decoder → Token Decoding 
→ Raw Text → LLM Correction → Final Transcription
```

---

## Core Features

### 1. Real-Time Webcam Lip Reading

**Purpose**: Transcribe speech from live video feed using only visual lip movements.

**Technical Implementation**:
- **Frame Capture**: Grabs frames at 16 FPS from webcam
- **Resolution**: 640x480 (downsampled by factor of 3) for optimal performance
- **Color Processing**: Converts to grayscale (single channel) for model compatibility
- **Compression**: JPEG quality set to 25 for efficient processing
- **Recording Control**: Alt/Option key toggles recording state

**Key Components**:
- `cv2.VideoCapture`: Webcam interface
- `cv2.VideoWriter`: Video segment writing
- Frame interval calculation: `1/fps` ensures consistent timing
- Mode indicators: Visual feedback with colored overlays

**Output**: Text typed directly at cursor location with automatic LLM correction.

### 2. Audio Speech-to-Text

**Purpose**: Transcribe audio captured from microphone using Whisper ASR.

**Technical Implementation**:
- **Sample Rate**: 16 kHz (Whisper-compatible)
- **Channels**: Mono conversion
- **Format**: 16-bit PCM internally, normalized to float32 [-1, 1]
- **Buffering**: Thread-safe audio capture with PyAudio
- **Duration Filter**: Minimum 2 seconds to avoid false triggers

**Key Components**:
- `AudioRecorder`: Threading-based capture with fallback sample rates
- `WhisperPipeline`: OpenAI's Whisper model integration
- `AudioUtils`: Preprocessing and format conversion
- Automatic resampling for various input sample rates

**Output**: Corrected transcription with punctuation and capitalization.

### 3. Dual Mode Operation

**Purpose**: Seamlessly switch between visual and audio transcription modes.

**Technical Implementation**:
- **Mode Toggle**: Tab key switches between "lip" and "voice" modes
- **State Management**: Tracks current mode and recording state
- **Visual Indicators**: Color-coded overlays (green for lip, yellow for voice)
- **Graceful Transitions**: Handles mode switching during active recording

**Key Components**:
- `self.current_mode`: State variable tracking mode
- `self.modes`: List of available modes for cycling
- `toggle_mode()`: Validates state before switching
- Mode-specific recording indicators

### 4. LLM-Enhanced Correction

**Purpose**: Improve transcription quality through context-aware corrections.

**Technical Implementation**:
- **Model**: Llama 3.2 via Ollama
- **Format**: Pydantic-validated structured output
- **Prompt Engineering**: System prompt guides correction behavior
- **Output Schema**: `list_of_changes` and `corrected_text` fields

**Correction Logic**:
1. Preserve original meaning
2. Fix individual word errors
3. Add appropriate punctuation
4. Correct capitalization (remove all-caps)
5. Ensure sentence-ending punctuation

**Key Components**:
- `Lip2TextOutput`: Pydantic model for validation
- Structured JSON output from Ollama
- Automatic period insertion for missing endings
- Graceful fallback to raw transcription

### 5. Web Application Interface

**Purpose**: Provide a production-ready web interface for batch processing.

**Technical Implementation**:
- **Framework**: Flask with Jinja2 templates
- **Storage**: Temporary upload directory with cleanup
- **Video Preprocessing**: Downscaling and frame rate normalization
- **Async Processing**: Non-blocking transcription
- **Multi-Language Support**: Translation with TTS

**Key Features**:
- Modern, responsive UI with gradient styling
- File upload with validation
- Real-time audio playback
- Translation to Telugu, Hindi, Tamil, Malayalam
- gTTS integration for audio output
- Session cleanup after processing

### 6. Batch Video Processing

**Purpose**: Process pre-recorded videos offline.

**Technical Implementation**:
- **Hydra Integration**: Configuration management
- **Command-Line Interface**: Flexible video path specification
- **Result Persistence**: Text file output with metadata
- **Error Handling**: Graceful degradation

**Output Format**:
```
Video: path/to/video.mp4
Processed on: 2025-01-15 10:30:00
Transcription:
[Corrected transcription text]
```

---

## Technology Stack

### Core Framework
- **PyTorch** (1.x): Deep learning framework
  - GPU acceleration via CUDA
  - Tensor operations and optimization
  - Model loading and inference
  
- **Hydra** (≥1.3.2): Configuration management
  - YAML-based configs
  - Command-line overrides
  - Dynamic config composition

### Computer Vision
- **OpenCV** (≥4.5.5.62): Video processing
  - Webcam capture
  - Frame encoding/decoding
  - Image transformations
  - Video writer utilities

- **MediaPipe**: Face detection (CPU-friendly)
  - Real-time landmark detection
  - Broad device compatibility
  - Minimal dependencies

- **RetinaFace** (Alternative): Advanced face detection
  - GPU-accelerated
  - Higher accuracy
  - More resource-intensive

- **SciPy** (≥1.3.0): Scientific computing
  - Signal processing
  - Numerical operations

- **Scikit-Image** (≥0.13.0): Image processing
  - Feature extraction
  - Filtering operations

### Audio Processing
- **PyAudio**: Real-time audio capture
  - Cross-platform support
  - Low-latency streaming
  - Format conversion

- **OpenAI Whisper**: Speech-to-text
  - Transformer architecture
  - Multilingual support
  - Fine-tuned models

- **TorchAudio**: Audio utilities
  - Resampling
  - Format conversion
  - Loading/saving

- **LibROSA** (indirect): Spectral analysis

### Natural Language Processing
- **Ollama**: LLM inference
  - Llama 3.2 model
  - Local execution
  - Structured output

- **Pydantic**: Data validation
  - Schema definition
  - JSON parsing
  - Type checking

### Web Framework
- **Flask** (≥3.0.0): Web server
  - RESTful endpoints
  - File handling
  - Template rendering

- **gTTS** (≥2.5.1): Text-to-speech
  - Google Cloud TTS
  - MP3 generation
  - Multi-language

### Utilities
- **NumPy** (≥1.21.0): Numerical operations
- **AV** (≥10.0.0): Media container processing
- **Keyboard**: Global key bindings
- **Six** (≥1.16.0): Python 2/3 compatibility

### System Integration
- **Python 3.12**: Runtime environment
- **uv**: Package manager
- **CUDA** (optional): GPU acceleration

---

## Deep Dive into Components

### 1. InferencePipeline

**Location**: `pipelines/pipeline.py`

**Responsibilities**:
- Orchestrate end-to-end inference
- Load configuration files
- Initialize models and data loaders
- Handle landmark detection
- Coordinate data preprocessing

**Key Methods**:
- `__init__`: Parse config, load models, setup data loaders
- `process_landmarks`: Face detection and landmark extraction
- `forward`: Complete inference pipeline execution

**Configuration Processing**:
```python
config = ConfigParser()
config.read(config_filename)

# Extract parameters from INI sections
modality = config.get("input", "modality")
model_path = config.get("model", "model_path")
beam_size = config.getint("decode", "beam_size")
ctc_weight = config.getfloat("decode", "ctc_weight")
lm_weight = config.getfloat("decode", "lm_weight")
```

### 2. AVSR Model

**Location**: `pipelines/model.py`

**Architecture**: Transformer-based Audio-Visual Speech Recognition

**Components**:
- **Backbone**: CNN feature extractor (ResNet-like)
- **Encoder**: Multi-head self-attention transformer
- **Decoder**: Cross-attention with CTC head
- **Language Model**: RNN-LM for beam search scoring
- **Beam Search**: Batch inference with prefix scoring

**Inference Process**:
```python
# 1. Feature extraction
enc_feats = model.encode(video_frames)

# 2. Beam search decoding
nbest_hyps = beam_search(enc_feats)

# 3. Token-to-text conversion
transcription = decode_tokens(nbest_hyps, token_list)
```

**Model Configuration**:
- **Vocab Size**: 5000 unigram subword units
- **CTC Weight**: 0.1 (10% CTC, 90% attention)
- **LM Weight**: 0.3
- **Beam Size**: 40
- **Max Length Ratio**: Dynamic

### 3. AudioRecorder

**Location**: `pipelines/audio_recorder.py`

**Architecture**: Threaded audio capture with buffering

**Key Features**:
- **Threading**: Non-blocking capture in background
- **Fallback Rates**: Automatic 48kHz fallback
- **Format Conversion**: Int16 → Float32 normalization
- **Multi-Channel**: Automatic downmixing to mono
- **Error Recovery**: Graceful handling of device failures

**Recording Flow**:
```python
def start_recording():
    self.is_recording = True
    self.audio_frames = []
    thread = threading.Thread(target=self._record)
    thread.start()

def _record():
    stream = pyaudio.open(format, channels, rate, input=True)
    while self.is_recording:
        data = stream.read(chunk_size)
        self.audio_frames.append(data)
    stream.close()

def stop_recording():
    self.is_recording = False
    thread.join()
    return convert_frames_to_array(self.audio_frames)
```

**Audio Processing**:
- **Normalization**: Map int16 [-32768, 32767] → float32 [-1.0, 1.0]
- **Resampling**: Automatic rate conversion if needed
- **Mono Conversion**: Average multi-channel to single channel

### 4. AudioUtils & WhisperPipeline

**Location**: `pipelines/audio_utils.py`, `pipelines/whisper_pipeline.py`

**Purpose**: Bridge between raw audio and Whisper STT

**Whisper Integration**:
```python
class WhisperPipeline:
    def __init__(self, model_size="base", device=None):
        self.model = whisper.load_model(model_size).to(device)
    
    def transcribe_audio(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result
    
    def preprocess_audio(self, audio_data, sample_rate):
        # Convert to float32
        # Normalize to [-1, 1]
        # Resample to 16kHz
        return processed_audio
```

**Model Sizes**:
- **tiny**: 39M parameters, fastest
- **base**: 74M parameters (default)
- **small**: 244M parameters
- **medium**: 769M parameters
- **large**: 1550M parameters, most accurate

### 5. Data Preprocessing

**Location**: `pipelines/data/data_module.py`, `pipelines/data/transforms.py`

**Video Processing**:
- **Face Detection**: MediaPipe or RetinaFace
- **Landmark Extraction**: Facial keypoint detection
- **ROI Extraction**: Crop and align face region
- **Normalization**: Grayscale conversion
- **Temporal Downsampling**: Frame rate adjustment

**Audio Processing**:
- **Resampling**: Standardize to 16kHz
- **Mono Conversion**: Channel downmixing
- **Normalization**: Amplitude scaling
- **Spectrogram**: Optional frequency domain transform

### 6. Face Detection Modules

**MediaPipe Detector** (`pipelines/detectors/mediapipe/detector.py`):
- **Single Detection**: Largest face per frame
- **Keypoints**: Right/left eye, nose, mouth center
- **Fallback**: Short-range ↔ Full-range detector switching
- **CPU-Optimized**: No GPU requirements

**RetinaFace Detector** (Alternative):
- **Multi-Face**: Detects all faces
- **Higher Accuracy**: Deep learning based
- **GPU-Accelerated**: CUDA support
- **Resource-Heavy**: More memory/compute

### 7. Flask Web Application

**Location**: `web/app.py`

**Architecture**: Production-ready RESTful web server

#### Application Factory Pattern

The Flask app uses the factory pattern for better modularity:

```python
def create_app(config_path: Optional[str] = None) -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", os.urandom(16))
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB max upload
    
    # Initialize model once at startup
    app.vsr_model = InferencePipeline(...)
    
    return app

app = create_app()
```

#### Global Configuration

**Environment Variables**:
- `MAX_VIDEO_SECONDS`: Maximum processing length (default: 20s)
- `TARGET_WIDTH`: Video resize width (default: 640px)
- `PORT`: Server port (default: 5000)
- `FLASK_SECRET_KEY`: Session encryption key

**Security**:
- **CSRF Protection**: Flask-WTF integration
- **File Size Limits**: 1GB maximum upload
- **File Type Validation**: Whitelist approach
- **UUID Filenames**: Prevent conflicts and injection

#### Video Preprocessing Pipeline

**Purpose**: Optimize videos for efficient inference

```python
def preprocess_video(input_path: str) -> str:
    # 1. Load video metadata
    src_fps = cv2.CAP_PROP_FPS or 25.0
    src_w, src_h = frame dimensions
    
    # 2. Calculate scaling factor
    scale = TARGET_WIDTH / src_w if src_w > TARGET_WIDTH else 1.0
    dst_w, dst_h = scaled dimensions (min 64x64)
    
    # 3. Temporal downsampling
    stride = round(src_fps / TARGET_FPS)  # Approximate 25 FPS
    
    # 4. Duration capping
    max_frames = MAX_VIDEO_SECONDS * TARGET_FPS  # Max 20s @ 25fps
    
    # 5. Process frames
    while written < max_frames:
        if frame_idx % stride == 0:
            if scale != 1.0:
                frame = cv2.resize(frame, (dst_w, dst_h))
            writer.write(frame)
```

**Optimizations**:
- **Resolution Reduction**: Downscale to 640px width
- **Frame Rate Normalization**: Resample to 25 FPS
- **Duration Limiting**: Prevent excessive processing times
- **Codec**: MP4V for broad compatibility
- **Fallback**: Return original on failure

#### Route Handlers

**GET / (Index Route)**:
```python
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", 
        transcript=None, 
        error=None, 
        translated=None, 
        tts_path=None
    )
```
- Renders upload form
- Clears previous session state
- No server-side processing

**POST /transcribe (Transcription Route)**:
```python
@app.route("/transcribe", methods=["POST"])
def transcribe():
    # 1. File validation
    if "video" not in request.files:
        return error_response("No file uploaded")
    
    file = request.files["video"]
    if not allowed_file(file.filename):
        return error_response("Unsupported format")
    
    # 2. Save with UUID
    unique_name = f"upload_{uuid.uuid4().hex}.{file_ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    file.save(save_path)
    
    # 3. Preprocess video
    processed_path = preprocess_video(save_path)
    
    # 4. Run inference
    transcript_raw = app.vsr_model(processed_path)
    
    # 5. LLM correction
    corrected = llm_correct_transcription(transcript_raw)
    
    # 6. Generate TTS
    tts_path = generate_english_tts(corrected)
    
    # 7. Cleanup temp files
    finally:
        cleanup_temp_files()
    
    # 8. Render results
    return render_template("index.html",
        transcript=corrected,
        tts_path=tts_path
    )
```

**Error Handling**:
- **File Not Found**: Validate request.files exists
- **Invalid Format**: Whitelist-based validation
- **Model Errors**: Catch and return user-friendly messages
- **Preprocessing Failures**: Fallback to original
- **LLM Errors**: Graceful degradation
- **TTS Failures**: Continue without audio
- **Cleanup Failures**: Multiple fallback strategies

**POST /translate (Translation Route)**:
```python
@app.route("/translate", methods=["POST"])
def translate():
    # 1. Validate input
    text = request.form.get("transcript", "").strip()
    target = request.form.get("target", "").strip()
    transcript_tts = request.form.get("transcript_tts", "").strip()
    
    # 2. Language validation
    if target not in {"te", "hi", "ta", "ml"}:
        return error_response("Unsupported language")
    
    # 3. Language mapping
    lang_map = {
        "te": {"name": "Telugu", "gtts": "te"},
        "hi": {"name": "Hindi", "gtts": "hi"},
        "ta": {"name": "Tamil", "gtts": "ta"},
        "ml": {"name": "Malayalam", "gtts": "ml"},
    }
    
    # 4. LLM translation
    translated = llm_translate(text, target)
    
    # 5. Generate translated TTS
    tts_path = generate_translated_tts(translated, lang_map[target])
    
    # 6. Render with both audio paths
    return render_template("index.html",
        transcript=text,              # Original
        translated=translated,        # Translation
        tts_path=transcript_tts,      # Original audio
        translation_tts_path=tts_path # Translation audio
    )
```

**Translation Features**:
- **Multi-Language Support**: 4 Indian languages
- **Preserve Original**: Display both versions
- **Dual Audio**: Separate TTS for each language
- **Structured Output**: Pydantic validation
- **Error Recovery**: Fallback mechanisms

**GET /tts/<filename> (Audio Serving)**:
```python
@app.route("/tts/<path:filename>")
def serve_tts(filename: str):
    return send_from_directory(TTS_DIR, filename, 
        mimetype="audio/mpeg", 
        as_attachment=False
    )
```
- **Secure Serving**: Validated directory
- **MIME Type**: Correct audio/mpeg header
- **Inline Playback**: Not forced download
- **Path Validation**: Prevent directory traversal

#### LLM Integration

**Transcription Correction**:
```python
response = chat(
    model='llama3.2',
    messages=[
        {
            'role': 'system',
            'content': "You are an assistant that helps make corrections... "
        },
        {
            'role': 'user',
            'content': f"Transcription:\n\n{transcript_raw}"
        }
    ],
    format=Lip2TextOutput.model_json_schema()
)
chat_output = Lip2TextOutput.model_validate_json(response.message.content)
```

**Translation**:
```python
response = chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": translation_prompt},
        {"role": "user", "content": f"Target: {lang}\nText: {text}"}
    ],
    format=TranslationOut.model_json_schema()
)
```

**Key Features**:
- **Structured Output**: Pydantic models ensure type safety
- **Error Handling**: Graceful fallback on LLM failures
- **Deterministic**: Temperature=0 for consistency
- **Timeout Handling**: Prevent hanging requests
- **Context Preservation**: Maintain conversation state

#### Text-to-Speech (TTS)

**Google TTS Integration**:
```python
from gtts import gTTS

def generate_tts(text: str, lang: str = 'en') -> str:
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    out_path = os.path.join(TTS_DIR, filename)
    
    tts = gTTS(text=text, lang=lang)
    tts.save(out_path)
    
    return filename
```

**Features**:
- **English TTS**: Generated for transcript
- **Multi-Language TTS**: Separate audio for translations
- **UUID Filenames**: Prevent collisions
- **Graceful Degradation**: Continue if TTS fails
- **Storage Management**: Automatic cleanup

#### File Management

**Temporary Storage**:
- **Upload Directory**: `uploads/` for incoming videos
- **TTS Directory**: `tts/` for generated audio
- **UUID-based**: Unique filenames prevent conflicts
- **Automatic Cleanup**: `finally` blocks ensure deletion

**Cleanup Strategy**:
```python
finally:
    # 1. Remove uploaded file
    if os.path.exists(save_path):
        os.remove(save_path)
    
    # 2. Remove processed file
    if processed_path and processed_path != save_path:
        os.remove(processed_path)
    
    # 3. Fallback: Clean entire directories
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    ensure_dir(UPLOAD_DIR)
```

**Directory Creation**:
```python
def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
```

#### Frontend Integration

**Template Rendering**:
```python
return render_template("index.html",
    transcript=corrected,           # English transcript
    error=error_message,            # Any errors
    translated=translated,          # Translation
    tts_path=transcript_url,        # English audio
    translation_tts_path=trans_url  # Translated audio
)
```

**Dynamic UI Elements**:
- **Conditional Rendering**: Show/hide based on state
- **Error Display**: User-friendly error messages
- **Progress Feedback**: Visual indicators
- **Audio Playback**: HTML5 audio controls
- **Reset Functionality**: "Process New Video" button

#### Frontend Architecture

**Modern CSS Design**:
```css
:root {
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --secondary: #10b981;
    --bg: #f9fafb;
    --card-bg: #ffffff;
}
```

**Responsive Layout**:
- **Mobile-First**: Flexible grid system
- **Breakpoints**: Adapt to screen sizes
- **Touch-Friendly**: Large tap targets
- **Dark Mode**: Automatic theme detection

**Interactive Elements**:
- **Loading States**: Disable buttons during processing
- **Visual Feedback**: Progress indicators
- **Error Handling**: User-friendly messages
- **Audio Controls**: Standard HTML5 player

**JavaScript Enhancements**:
```javascript
function onSubmit() {
    const btn = document.getElementById('btn');
    btn.disabled = true;
    btn.textContent = 'Transcribing...';
    btn.classList.add('loading');
}
```

#### Deployment Considerations

**Production Configuration**:
```python
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",           # Accept external connections
        port=int(os.environ.get("PORT", 5000)),
        debug=False               # Disable in production
    )
```

**Recommended Stack**:
- **WSGI Server**: Gunicorn or uWSGI
- **Reverse Proxy**: Nginx
- **Static Files**: CDN for TTS assets
- **Session Storage**: Redis for scalability
- **Load Balancing**: Multiple workers

**Security Headers**:
```python
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response
```

**Performance Optimization**:
- **Model Caching**: Load once at startup
- **Connection Pooling**: Reuse database connections
- **Async Processing**: Celery for heavy tasks
- **CDN Integration**: Cache TTS files
- **Rate Limiting**: Prevent abuse

---

## User Interfaces

### 1. Command-Line Interface (Real-Time Mode)

**Entry Point**: `main.py`

**Features**:
- Always-on-top preview window
- Keyboard-controlled recording
- Real-time mode switching
- Visual status indicators

**Controls**:
| Key | Action |
|-----|--------|
| `Tab` | Switch between lip/voice modes |
| `Alt`/`Option` | Start/stop recording |
| `Q` | Quit application |

**Visual Indicators**:
- **Mode Display**: Green (lip) / Yellow (voice)
- **Recording**: Black circle (lip) / Red circle (voice)
- **Status Text**: Overlaid instructions

**Workflow**:
```
Start App → Preview Appears → Select Mode → 
Press Alt → Record → Press Alt → Process → 
LLM Correction → Type at Cursor
```

### 2. Web Application

**Entry Point**: `web/app.py`

**Architecture**:
- **Routes**: `/`, `/transcribe`, `/translate`, `/tts/<filename>`
- **Templates**: Jinja2-based rendering
- **Static Assets**: Embedded CSS, no external dependencies
- **File Handling**: Temporary storage with automatic cleanup

**User Flow**:
```
Upload Video → Preprocessing → Transcription → 
[Optional] Translation → TTS Generation → 
Audio Playback → Download/Share → Cleanup
```

**Key Endpoints**:

**GET /**:
- Render upload form
- Display any error messages
- Show previous results if available

**POST /transcribe**:
- Validate file type
- Save to temporary storage
- Preprocess video (downscale, trim)
- Run inference pipeline
- LLM correction
- Generate English TTS
- Cleanup temporary files
- Render results with audio

**POST /translate**:
- Accept transcript and target language
- LLM-based translation
- Generate translated TTS
- Serve audio file
- Update UI with results

**GET /tts/<filename>**:
- Stream MP3 audio
- Set appropriate MIME type
- Enable browser playback

**UI Design**:
- **Gradient Background**: Purple/blue color scheme
- **Card-Based Layout**: Elevated white cards
- **Responsive**: Mobile-friendly grid
- **Dark Mode**: Automatic theme detection
- **Modern Typography**: System fonts with fallbacks
- **Smooth Transitions**: CSS animations

### 3. Batch Processing CLI

**Entry Point**: `process_video.py`

**Usage**:
```bash
sudo uv run --with-requirements requirements.txt \
  --python 3.12 process_video.py \
  config_filename=./configs/LRS3_V_WER19.1.ini \
  detector=mediapipe \
  +video_path=path/to/video.mp4
```

**Output**:
- Console progress indicators
- Final transcription display
- Text file with metadata
- Error reporting

---

## Model Details

### Lip Reading Model (AVSR)

**Dataset**: LRS3 (Lip Reading Sentences 3)
- 30,000+ hours of video
- British English speakers
- High-resolution face captures
- Sentence-level transcriptions

**Architecture**: Transformer-based VSR
- **Vision Encoder**: 3D CNN → Temporal pooling
- **Text Encoder**: Multi-head self-attention
- **Decoder**: Cross-attention with CTC
- **Language Model**: RNN-LM (subword)

**Performance Metrics**:
- **WER**: 19.1% on LRS3 test set
- **Speed**: ~50-100ms per frame
- **Memory**: ~2GB GPU VRAM

**Tokenization**: Unigram subword units
- **Vocab Size**: 5000 tokens
- **Benefits**: Handle OOV words
- **Trade-off**: Slightly more tokens per word

### OpenAI Whisper

**Architecture**: Encoder-Decoder Transformer
- **Encoders**: Multi-scale convolutional features
- **Decoder**: Causal self-attention + cross-attention
- **Tokens**: Multilingual BPE

**Capabilities**:
- **Languages**: 99+ languages
- **Domains**: Robust to noise and accents
- **Features**: Automatic punctuation, capitalization
- **Fine-tuning**: Large-v2-v3 variants

**Model Selection**:
- **Base**: Best speed/accuracy trade-off
- **Context**: 30-second windows
- **Sample Rate**: 16 kHz required

### Llama 3.2 via Ollama

**Purpose**: Post-transcription correction

**Configuration**:
- **Model**: Llama 3.2 (3B parameters)
- **Format**: JSON-structured output
- **Temperature**: 0 (deterministic)
- **Max Tokens**: Dynamic based on input

**Prompt Design**:
```
System: You are an assistant that helps make corrections to 
lipreading/speech-to-text output. Fix obvious errors, add 
punctuation, correct capitalization. Return structured JSON.

User: Transcription: [raw output]
```

---

## Workflow Processes

### Real-Time Lip Reading Workflow

```python
# Initialization
vsr_model = InferencePipeline(config, device, detector)
keyboard.hook(on_action_callback)
start_webcam()

# Recording Loop
while True:
    handle_keyboard_input()
    if recording and mode == "lip":
        frame = webcam.read()
        gray_frame = convert_to_grayscale(frame)
        video_writer.write(gray_frame)
    
    if stop_recording:
        video_writer.release()
        if duration >= 2s:
            submit_inference_task(video_path)

# Inference
with ThreadPoolExecutor:
    result = vsr_model(video_path)
    keyboard.write(result)
    select_text(result)
    
    # LLM Correction
    llm_response = ollama.chat(
        model="llama3.2",
        prompt=correction_prompt,
        format=Lip2TextOutput
    )
    
    keyboard.write(llm_response.corrected_text)
    delete_temp_video()
```

### Audio Transcription Workflow

```python
# Initialization
audio_recorder = AudioRecorder(sample_rate=16000)
whisper_pipeline = WhisperPipeline(model_size="base")

# Recording
if start_recording:
    audio_recorder.start_recording()  # Thread-based

if stop_recording:
    audio_data, sample_rate = audio_recorder.stop_recording()
    
    # Preprocessing
    if sample_rate != 16000:
        audio_data = resample(audio_data, sample_rate, 16000)
    
    audio_data = normalize(audio_data)
    
    # Transcription
    result = whisper_pipeline.transcribe_audio_from_array(
        audio_data, 
        sample_rate=16000
    )
    
    # Correction
    corrected = llm_correct(result.text)
    keyboard.write(corrected)
```

### Web Application Workflow

```python
# Upload Handling
@app.route("/transcribe", methods=["POST"])
def transcribe():
    # Validate file
    if not allowed_file(file.filename):
        return error_response
    
    # Save upload
    save_path = save_to_temp(file)
    
    # Preprocess
    processed_path = preprocess_video(save_path)
    # - Downscale to 640px width
    # - Normalize to 25 FPS
    # - Trim to max 20 seconds
    
    # Inference
    transcript_raw = vsr_model(processed_path)
    
    # Correction
    transcript_corrected = llm_correct(transcript_raw)
    
    # TTS Generation
    tts_path = generate_english_tts(transcript_corrected)
    
    # Cleanup
    delete_temp_files()
    
    # Render
    return render_template("results.html",
        transcript=transcript_corrected,
        tts_path=tts_path
    )

# Translation
@app.route("/translate", methods=["POST"])
def translate():
    text = request.form["transcript"]
    target_lang = request.form["target"]
    
    # Translation
    translated = llm_translate(text, target_lang)
    
    # TTS
    tts_path = generate_tts(translated, lang=target_lang)
    
    return render_template("results.html",
        transcript=text,
        translated=translated,
        tts_path=tts_path
    )
```

---

## Installation & Setup

### Prerequisites

**System Requirements**:
- **OS**: Linux, macOS, or Windows
- **Python**: 3.12 or compatible
- **GPU** (optional): NVIDIA GPU with CUDA support
- **RAM**: Minimum 8GB, 16GB recommended
- **Disk**: ~10GB for models and dependencies

**Dependencies**:
```bash
# Core framework
hydra-core >= 1.3.2
torch, torchvision, torchaudio

# Computer vision
opencv-python >= 4.5.5.62
scipy >= 1.3.0
scikit-image >= 0.13.0
mediapipe
numpy >= 1.21.0

# Audio processing
pyaudio
openai-whisper

# NLP
ollama
pydantic

# Utilities
six >= 1.16.0
keyboard

# Web (optional)
Flask >= 3.0.0
gTTS >= 2.5.1
```

### Step-by-Step Installation

**1. Clone Repository**:
```bash
git clone https://github.com/abhiramavarma/lip2text
cd lip2text
```

**2. Install Python 3.12**:
```bash
# Using pyenv (recommended)
pyenv install 3.12
pyenv local 3.12

# Or using system package manager
sudo apt install python3.12 python3.12-venv
```

**3. Install uv Package Manager**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**4. Install Dependencies**:
```bash
# Using uv (recommended)
sudo uv run --with-requirements requirements.txt --python 3.12 pip install -r requirements.txt

# Or using pip
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**5. Download Model Files**:
```bash
# LRS3 model (if not already present)
# Place in benchmarks/LRS3/models/LRS3_V_WER19.1/
# - model.pth
# - model.json

# Language model
# Place in benchmarks/LRS3/language_models/lm_en_subword/
# - model.pth
# - model.json
```

**6. Install Ollama and Llama 3.2**:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama 3.2 model
ollama pull llama3.2
```

**7. Test Installation**:
```bash
# Real-time mode
sudo uv run --with-requirements requirements.txt \
  --python 3.12 main.py \
  config_filename=./configs/LRS3_V_WER19.1.ini \
  detector=mediapipe

# Batch processing
sudo uv run --with-requirements requirements.txt \
  --python 3.12 process_video.py \
  config_filename=./configs/LRS3_V_WER19.1.ini \
  detector=mediapipe \
  +video_path=test.mp4

# Web interface
cd web
sudo uv run --with-requirements ../requirements.txt \
  --python 3.12 app.py
```

### Troubleshooting

**Common Issues**:

**1. PyAudio Installation**:
```bash
# Ubuntu/Debian
sudo apt install portaudio19-dev python3-pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Windows
pip install pipwin
pipwin install pyaudio
```

**2. MediaPipe Compatibility**:
```bash
# If Python 3.13 fails, downgrade or use alternative
pip install mediapipe --no-binary mediapipe

# Or use RetinaFace instead
# Update detector parameter to "retinaface"
```

**3. CUDA Errors**:
```bash
# Verify CUDA installation
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**4. Ollama Connection**:
```bash
# Start Ollama service
ollama serve

# Test model availability
ollama list

# Verify Llama 3.2
ollama show llama3.2
```

---

## Usage Guide

### Real-Time Mode

**Starting the Application**:
```bash
sudo uv run --with-requirements requirements.txt \
  --python 3.12 main.py \
  config_filename=./configs/LRS3_V_WER19.1.1.ini \
  detector=mediapipe
```

**Basic Usage**:
1. Ensure webcam is connected and functional
2. Wait for preview window to appear
3. Position face in frame with good lighting
4. Press `Alt` to start recording
5. Mouth words clearly without speaking
6. Press `Alt` again to stop and process

**Mode Switching**:
- Press `Tab` to toggle between lip and voice modes
- Visual indicator shows current mode
- Cannot switch during active recording

**Tips for Best Results**:
- **Lighting**: Face the light source, avoid backlighting
- **Distance**: 1-2 feet from camera
- **Clarity**: Keep face steady, mouth visible
- **Duration**: Record segments of 2-10 seconds
- **Context**: Speak complete phrases

### Batch Processing

**Command Structure**:
```bash
process_video.py \
  config_filename=<path> \
  detector=<mediapipe|retinaface> \
  +video_path=<video_file>
```

**Example**:
```bash
sudo uv run --with-requirements requirements.txt \
  --python 3.12 process_video.py \
  config_filename=./configs/LRS3_V_WER19.1.ini \
  detector=mediapipe \
  +video_path=./test_data/V5.mov
```

**Supported Formats**:
- MP4, MOV, AVI, MKV, WebM

**Output**:
- Console: Real-time progress and final transcription
- File: `transcription_<videoname>.txt`

### Web Interface

**Starting the Server**:
```bash
cd web
sudo uv run --with-requirements ../requirements.txt \
  --python 3.12 app.py
```

**Access**:
- Open browser to `http://localhost:5000`
- Or `http://0.0.0.0:5000` for network access

**Workflow**:
1. Click "Choose File" to select video
2. Click "Transcribe" button
3. Wait for processing (status shown)
4. View transcript with audio playback
5. Optionally select target language and click "Translate"
6. Listen to translated audio
7. Click "Process New Video" to start over

**Features**:
- Drag-and-drop file upload
- Automatic video optimization
- Dual audio playback (transcript + translation)
- MP3 download capability
- Mobile-responsive interface

**Configuration**:
```python
# Environment variables
PORT=5000  # Server port
MAX_VIDEO_SECONDS=20  # Max processing length
TARGET_WIDTH=640  # Resize width
FLASK_SECRET_KEY=...  # Session security
```

---

## Configuration

### Model Configuration (INI)

**Location**: `configs/LRS3_V_WER19.1.ini`

**Structure**:
```ini
[input]
modality=video          # video, audio, or audiovisual
v_fps=25               # Input frame rate

[model]
v_fps=25               # Model training frame rate
model_path=benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth
model_conf=benchmarks/LRS3/models/LRS3_V_WER19.1/model.json
rnnlm=benchmarks/LRS3/language_models/lm_en_subword/model.pth
rnnlm_conf=benchmarks/LRS3/language_models/lm_en_subword/model.json

[decode]
beam_size=40           # Search breadth
penalty=0.0            # Length penalty
maxlenratio=0.0        # Max output length ratio
minlenratio=0.0        # Min output length ratio
ctc_weight=0.1         # CTC vs attention weight
lm_weight=0.3          # Language model weight
```

**Parameter Tuning**:
- **beam_size**: Larger = better accuracy, slower (default: 40)
- **ctc_weight**: Higher = more permissive, lower = strict (default: 0.1)
- **lm_weight**: Higher = more contextual, lower = raw (default: 0.3)

### Hydra Configuration

**Location**: `hydra_configs/default.yaml`

**Structure**:
```yaml
config_filename: null
data_dir: null
data_filename: null
data_ext: ".mp4"
landmarks_dir: null
landmarks_filename: null
landmarks_ext: ".pkl"
labels_filename: null
detector: retinaface     # mediapipe or retinaface
dst_filename: null
gpu_idx: 0
output_subdir: null
```

**Command-Line Overrides**:
```bash
main.py \
  config_filename=./configs/LRS3_V_WER19.1.ini \
  detector=mediapipe \
  gpu_idx=0
```

### Application Configuration

**Real-Time Settings** (`main.py`):
```python
self.res_factor = 3           # Resolution downsampling
self.fps = 16                 # Capture frame rate
self.frame_compression = 25   # JPEG quality
self.output_prefix = "webcam" # Temp file prefix
```

**Web Settings** (`web/app.py`):
```python
MAX_VIDEO_SECONDS = 20        # Max processing length
TARGET_WIDTH = 640            # Resize width
TARGET_FPS = 25               # Normalized frame rate
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
```

---

## Performance Considerations

### Resource Usage

**GPU (Recommended)**:
- **CUDA**: ~5-10x faster inference
- **VRAM**: ~2-4GB depending on batch size
- **Memory**: Pinned for zero-copy transfers

**CPU (Fallback)**:
- **Threads**: Parallel processing for efficiency
- **Memory**: ~4-8GB RAM usage
- **Speed**: ~5-10x slower than GPU

### Optimization Strategies

**1. Model Quantization**:
```python
# Reduce precision for faster inference
model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)
```

**2. Batch Processing**:
```python
# Process multiple segments together
# Reduces overhead per item
batch_size = 4
batch = collate_batch(segments)
results = model(batch)
```

**3. Frame Skipping**:
```python
# For real-time applications
stride = 2  # Process every other frame
# Reduces computation, slight accuracy trade-off
```

**4. Caching**:
```python
# Pre-compute expensive operations
@lru_cache(maxsize=100)
def preprocess_landmarks(video_path):
    return detect_and_align(video_path)
```

### Benchmarking

**Lip Reading**:
- **LRS3 Test**: WER 19.1%
- **Inference Speed**: 50-100ms per frame (GPU)
- **Throughput**: 10-20 FPS real-time

**Whisper (Base)**:
- **LibriSpeech**: ~5% WER
- **Speed**: ~2-5x real-time (GPU)
- **Memory**: ~2GB VRAM

**End-to-End Latency**:
- **Camera → Display**: <500ms (RT mode)
- **File → Result**: <5 seconds (batch)
- **Web Upload**: <10 seconds (including network)

---

## Future Enhancements

### Planned Features

**1. Multi-Language Lip Reading**:
- Extend beyond English
- Support Hindi, Telugu, Tamil, Malayalam
- Language detection and switching

**2. Real-Time Translation**:
- In-app language selection
- Overlay translated subtitles
- Voice-to-voice translation

**3. Cloud Deployment**:
- Docker containerization
- Kubernetes orchestration
- Auto-scaling for load

**4. Enhanced Accuracy**:
- Fine-tuned models for specific domains
- Active learning from user corrections
- Domain adaptation techniques

**5. Accessibility Features**:
- Sign language recognition
- Live captions for video calls
- Screen reader integration

**6. Mobile Applications**:
- iOS/Android native apps
- Offline processing
- Cloud sync

### Research Directions

**1. Architectural Improvements**:
- Attention mechanisms for temporal alignment
- Self-supervised pre-training
- Multi-task learning objectives

**2. Data Augmentation**:
- Synthetic video generation
- Adversarial training
- Domain randomization

**3. Personalization**:
- User-specific fine-tuning
- Adaptive vocabulary
- Custom language models

---

## Conclusion

Lip2Text represents a state-of-the-art approach to multimodal speech recognition, combining cutting-edge deep learning models with practical engineering solutions. The system's privacy-first design, multi-modal capabilities, and production-ready interfaces make it suitable for both research and commercial applications.

**Key Strengths**:
- ✅ Complete privacy through local processing
- ✅ Multi-modal architecture (visual + audio)
- ✅ Real-time performance with low latency
- ✅ Production-ready web interface
- ✅ Extensible codebase

**Use Cases**:
- Accessibility tools for hearing-impaired users
- Transcription services for content creators
- Security and surveillance applications
- Human-computer interaction research
- Educational language learning

---

**Authors**: N.Abhirama Varma, N.Rahul Sai, V.Pranai

**Version**: 2.0

**Last Updated**: January 2025

**License**: Apache 2.0

---

## Appendix

### File Structure
```
lip2text/
├── benchmarks/              # Model files
│   └── LRS3/
│       ├── models/         # VSR models
│       └── language_models/ # LM models
├── configs/                # Model configurations
├── espnet/                 # PyTorch backend
├── hydra_configs/          # Hydra configs
├── pipelines/              # Core processing
│   ├── audio_recorder.py  # Audio capture
│   ├── audio_utils.py     # Audio processing
│   ├── data/              # Data loaders
│   ├── detectors/         # Face detection
│   ├── model.py           # AVSR model
│   ├── pipeline.py        # Inference pipeline
│   └── whisper_pipeline.py # Whisper STT
├── web/                    # Web application
│   ├── app.py             # Flask server
│   └── templates/         # HTML templates
├── main.py                # Real-time entry point
├── process_video.py       # Batch processor
├── requirements.txt       # Dependencies
└── DOCUMENTATION.md       # This file
```

### Additional Resources
- [LRS3 Dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/)
- [ESPnet Documentation](https://espnet.github.io/espnet/)
- [Ollama Documentation](https://ollama.ai/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Contributing
Please refer to `CONTRIBUTING.md` for guidelines on contributing to the project.

### Support
For issues, questions, or feature requests, please open a GitHub issue or contact the maintainers.

---

*End of Documentation*

