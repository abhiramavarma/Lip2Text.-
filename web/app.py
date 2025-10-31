import os
import sys
import uuid
import shutil
from typing import Optional

from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import torch
import cv2


# Ensure project root is on sys.path so we can import pipelines
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipelines.pipeline import InferencePipeline  # noqa: E402


# Configuration
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "LRS3_V_WER19.1.ini")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
TTS_DIR = os.path.join(PROJECT_ROOT, "tts")
MAX_VIDEO_SECONDS = int(os.environ.get("MAX_VIDEO_SECONDS", "20"))
TARGET_WIDTH = int(os.environ.get("TARGET_WIDTH", "640"))
TARGET_FPS = 25


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def create_app(config_path: Optional[str] = None) -> Flask:
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", os.urandom(16))
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

    ensure_dir(UPLOAD_DIR)
    ensure_dir(TTS_DIR)

    # Initialize model once
    model_cfg = config_path or DEFAULT_CONFIG_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use mediapipe detector for broad compatibility (CPU-friendly)
    app.vsr_model = InferencePipeline(
        model_cfg,
        detector="mediapipe",
        face_track=True,
        device=device,
    )

    def preprocess_video(input_path: str) -> str:
        """Trim and downscale the uploaded video to reduce memory usage for inference."""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return input_path

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            scale = 1.0
            if src_w > TARGET_WIDTH:
                scale = TARGET_WIDTH / float(src_w)
            dst_w = max(64, int(src_w * scale))
            dst_h = max(64, int(src_h * scale))

            # Compute sampling stride to approximate TARGET_FPS
            stride = max(1, int(round(src_fps / float(TARGET_FPS))))
            max_frames = MAX_VIDEO_SECONDS * TARGET_FPS

            processed_name = f"processed_{uuid.uuid4().hex}.mp4"
            processed_path = os.path.join(UPLOAD_DIR, processed_name)
            writer = cv2.VideoWriter(
                processed_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                TARGET_FPS,
                (dst_w, dst_h),
                True,
            )
            if not writer.isOpened():
                cap.release()
                return input_path

            frame_idx = 0
            written = 0
            while written < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % stride == 0:
                    if scale != 1.0:
                        frame = cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
                    writer.write(frame)
                    written += 1
                frame_idx += 1

            writer.release()
            cap.release()

            # If we could not write anything, fallback to original
            if written == 0:
                try:
                    os.remove(processed_path)
                except Exception:
                    pass
                return input_path

            return processed_path
        except Exception:
            # On any preprocessing failure, fallback to original
            try:
                cap.release()  # type: ignore[name-defined]
            except Exception:
                pass
            return input_path

    @app.route("/", methods=["GET"])  # type: ignore[misc]
    def index():
        return render_template("index.html", transcript=None, error=None, translated=None, tts_path=None)

    @app.route("/transcribe", methods=["POST"])  # type: ignore[misc]
    def transcribe():
        if "video" not in request.files:
            flash("No file part in the request")
            return redirect(url_for("index"))

        file = request.files["video"]
        if file.filename == "":
            return render_template("index.html", transcript=None, error="No file selected.")

        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                transcript=None,
                error=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        # Save with a unique name to avoid collisions
        file_ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"upload_{uuid.uuid4().hex}.{file_ext}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)
        file.save(save_path)

        processed_path = None
        try:
            # Preprocess video to cap duration/size for stability
            processed_path = preprocess_video(save_path)

            # Run transcription (raw, caps) on processed video
            transcript_raw = app.vsr_model(processed_path) or ""
            transcript_raw = transcript_raw.strip()

            # LLM correction to mirror main.py behavior
            corrected = transcript_raw
            if transcript_raw:
                from pydantic import BaseModel
                from ollama import chat

                class Lip2TextOutput(BaseModel):
                    list_of_changes: str
                    corrected_text: str

                response = chat(
                    model='llama3.2',
                    messages=[
                        {
                            'role': 'system',
                            'content': (
                                "You are an assistant that helps make corrections to the output of a lipreading model. "
                                "The text you will receive was transcribed using a video-to-text system that attempts to lipread the subject speaking in the video, so the text will likely be imperfect.\n\n"
                                "If something seems unusual, assume it was mistranscribed. Do your best to infer the words actually spoken, and make changes to the mistranscriptions in your response. "
                                "Do not add more words or content, just change the ones that seem to be out of place (and, therefore, mistranscribed). "
                                "Do not change even the wording of sentences, just individual words that look nonsensical in the context of all of the other words in the sentence.\n\n"
                                "Also, add correct punctuation to the entire text. ALWAYS end each sentence with the appropriate sentence ending: '.', '?', or '!'. "
                                "The input text in all-caps, although your response should be capitalized correctly and should NOT be in all-caps.\n\n"
                                "Return the corrected text in the format of 'list_of_changes' and 'corrected_text'."
                            )
                        },
                        {
                            'role': 'user',
                            'content': f"Transcription:\n\n{transcript_raw}"
                        }
                    ],
                    format=Lip2TextOutput.model_json_schema()
                )

                chat_output = Lip2TextOutput.model_validate_json(response.message.content)
                corrected = chat_output.corrected_text or transcript_raw
                if corrected and corrected[-1] not in ['.', '?', '!']:
                    corrected += '.'

            if not corrected:
                corrected = "No speech detected"

            # Generate TTS for the transcript (English)
            tts_rel = None
            try:
                from gtts import gTTS

                filename = f"tts_{uuid.uuid4().hex}.mp3"
                out_path = os.path.join(TTS_DIR, filename)
                tts = gTTS(text=corrected, lang='en')
                tts.save(out_path)
                tts_rel = filename
            except Exception as e:
                # Continue without audio if TTS fails
                pass

            tts_url = url_for("serve_tts", filename=tts_rel) if tts_rel else None
            return render_template("index.html", transcript=corrected, error=None, translated=None, tts_path=tts_url)
        except Exception as e:
            return render_template("index.html", transcript=None, error=str(e), translated=None, tts_path=None)
        finally:
            # Cleanup uploaded file
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
            except Exception:
                try:
                    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
                    ensure_dir(UPLOAD_DIR)
                except Exception:
                    pass
            # Cleanup processed file
            try:
                if processed_path and os.path.exists(processed_path) and processed_path != save_path:
                    os.remove(processed_path)
            except Exception:
                pass

    @app.route("/translate", methods=["POST"])  # type: ignore[misc]
    def translate():
        # Expect original corrected transcript, target language code, and transcript audio
        text = request.form.get("transcript", "").strip()
        target = request.form.get("target", "").strip()
        transcript_tts = request.form.get("transcript_tts", "").strip()

        if not text:
            return render_template("index.html", transcript=None, translated=None, tts_path=None, error="No transcript to translate.")
        if target not in {"te", "hi", "ta", "ml"}:
            return render_template("index.html", transcript=text, translated=None, tts_path=transcript_tts, error="Unsupported language.")

        # Map to human name and gTTS language codes
        lang_map = {
            "te": {"name": "Telugu", "gtts": "te"},
            "hi": {"name": "Hindi", "gtts": "hi"},
            "ta": {"name": "Tamil", "gtts": "ta"},
            "ml": {"name": "Malayalam", "gtts": "ml"},
        }

        # Use LLM to translate
        translated = None
        try:
            from pydantic import BaseModel
            from ollama import chat

            class TranslationOut(BaseModel):
                translated_text: str

            system_prompt = (
                "You are a translation assistant. Translate the user's text to the requested target language. "
                "Do not add explanations, only return the translation."
            )
            user_prompt = f"Target language: {lang_map[target]['name']}\n\nText:\n{text}"

            resp = chat(
                model="llama3.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=TranslationOut.model_json_schema(),
            )

            out = TranslationOut.model_validate_json(resp.message.content)
            translated = (out.translated_text or "").strip()
        except Exception as e:
            return render_template("index.html", transcript=text, translated=None, tts_path=transcript_tts, error=f"Translation failed: {e}")

        if not translated:
            return render_template("index.html", transcript=text, translated=None, tts_path=transcript_tts, error="Empty translation.")

        # Generate TTS using gTTS
        tts_rel = None
        try:
            from gtts import gTTS

            filename = f"tts_{uuid.uuid4().hex}.mp3"
            out_path = os.path.join(TTS_DIR, filename)
            tts = gTTS(text=translated, lang=lang_map[target]["gtts"])  # type: ignore[arg-type]
            tts.save(out_path)
            tts_rel = filename
        except Exception as e:
            # Continue without audio if TTS fails
            return render_template("index.html", transcript=text, translated=translated, tts_path=transcript_tts, error=f"TTS failed: {e}")

        translation_audio_url = url_for("serve_tts", filename=tts_rel) if tts_rel else None
        return render_template("index.html", transcript=text, translated=translated, tts_path=transcript_tts, translation_tts_path=translation_audio_url, error=None)

    @app.route("/tts/<path:filename>")  # type: ignore[misc]
    def serve_tts(filename: str):
        # Serve generated TTS files
        return send_from_directory(TTS_DIR, filename, mimetype="audio/mpeg", as_attachment=False)

    return app


app = create_app()


if __name__ == "__main__":
    # Run development server
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)


