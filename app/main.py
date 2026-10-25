"""
FastAPI Application and Local REST API for Personal AI Time Capsule.
Integrates Conversation Engine, Memory Vault, Audio Synthesis,
Avatar Rendering, Ingestion, and Backup/Disaster Recovery.
"""

from __future__ import annotations
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.models import (
    MemoryItem,
    TemporalBelief,
    MemoryCorrection,
    VerificationStatus,
    EmotionCategory,
    ExpressiveDeliveryParams
)
from core.config import CapsuleConfig, ensure_directories
from memory.database import DatabaseManager
from memory.vector_store import VectorStore
from memory.retrieval import HybridRetrievalProvider
from memory.backup import BackupManager
from core.personality import PersonalityProfile
from core.conversation import ConversationEngine
from core.voice import ExpressiveDeliveryController, VoiceProfile
from core.jev_engine import LocalJevEngine, DualModelVerifier, pick_best_two_models
from providers.llm.ollama_provider import OllamaLLMProvider
from providers.llm.mock_provider import MockLLMProvider
from providers.tts.macos_tts import MacOSNativeTTSProvider
from providers.avatar.liveportrait_provider import LivePortraitAvatarProvider
from providers.asr.whisper_provider import PyAVWhisperASRProvider
from ingestion.pipeline import MediaIngestionPipeline


# Initialize configuration and paths
config = CapsuleConfig.load_or_create()
ensure_directories(config)

app = FastAPI(
    title="Personal AI Time Capsule",
    description="Locally runnable private AI representation preserving personal memory, voice, and evolving identity.",
    version="1.0.0"
)

# Mount static files and templates
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "ui" / "static"
TEMPLATES_DIR = BASE_DIR / "ui" / "templates"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize Subsystems
db = DatabaseManager(config.storage.sqlite_db_path)
vector_store = VectorStore(config.storage.chroma_dir)
retrieval = HybridRetrievalProvider(db, vector_store)
backup_mgr = BackupManager(config, db)

# Select LLM Provider
if config.llm.provider == "ollama":
    llm = OllamaLLMProvider(config.llm)
    if not llm.is_healthy():
        print("[App] Ollama not responsive, using fallback Mock LLM Provider.")
        llm = MockLLMProvider()
else:
    llm = MockLLMProvider()

jev_verifier = None
if getattr(config.llm, "enable_jev_verification", True):
    primary_m, critic_m = pick_best_two_models(
        ollama_endpoint=config.llm.endpoint,
        mlx_endpoint=config.llm.mlx_endpoint
    )
    critic_engine = LocalJevEngine(
        model_name=critic_m,
        endpoint=config.llm.endpoint,
        mlx_endpoint=config.llm.mlx_endpoint,
        llm_provider=llm if isinstance(llm, MockLLMProvider) else None
    )
    jev_verifier = DualModelVerifier(
        critic_engine=critic_engine,
        primary_model=primary_m,
        critic_model=critic_m
    )
    retrieval.jev_engine = critic_engine

personality = PersonalityProfile()
conversation_engine = ConversationEngine(retrieval, db, llm, personality, jev_verifier=jev_verifier)

# Voice & TTS Provider
delivery_controller = ExpressiveDeliveryController()
if config.tts.provider in ["kokoro", "neural"]:
    fallback_tts = MacOSNativeTTSProvider(
        default_voice="Daniel",
        sample_rate=config.tts.sample_rate,
        delivery_controller=delivery_controller
    )
    from providers.tts.neural_tts import NeuralTTSProvider
    tts_provider = NeuralTTSProvider(
        model_name="kokoro-82m",
        voice_profile=delivery_controller.profile,
        fallback_provider=fallback_tts
    )
else:
    tts_provider = MacOSNativeTTSProvider(
        default_voice=config.tts.default_voice,
        sample_rate=config.tts.sample_rate,
        delivery_controller=delivery_controller
    )

# Avatar Provider
avatar_provider = LivePortraitAvatarProvider(
    output_dir=os.path.join(config.storage.data_dir, "avatar_cache"),
    target_fps=config.avatar.target_fps,
    resolution=config.avatar.resolution
)

# Media Ingestion Pipeline
ingestion_pipeline = MediaIngestionPipeline(config, db, vector_store)

# Local ASR Speech-to-Text Provider
asr_provider = PyAVWhisperASRProvider(model_size=config.asr.model_size)


# --- Pydantic Request Models ---

class ChatRequest(BaseModel):
    query: str
    era_id: Optional[str] = None
    verification_filter: Optional[str] = None
    synthesize_audio: bool = False
    render_avatar: bool = False
    synthesize_avatar: bool = False
    force_simulation: bool = False
    session_id: Optional[str] = None


class SessionCreateRequest(BaseModel):
    title: Optional[str] = "New Conversation"



class TTSRequest(BaseModel):
    text: str
    emotion: str = "neutral"
    intensity: float = 0.5
    speech_rate: float = 1.0
    pitch_shift: float = 0.0
    voice: Optional[str] = None


class MemoryCorrectionRequest(BaseModel):
    memory_id: str
    corrected_content: str
    correction_reason: str


class IngestionRequest(BaseModel):
    target_directory: Optional[str] = None
    is_synthetic_fixture: bool = False


# --- API Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "snapshots": personality.list_snapshots(),
            "stats": db.get_archive_stats()
        }
    )


@app.get("/api/health")
async def health_check():
    """System health check across all subcomponents."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": True,
        "vector_store": vector_store.is_available,
        "llm_provider": {
            "name": config.llm.provider,
            "healthy": llm.is_healthy(),
            "info": llm.get_model_info()
        },
        "tts_provider": {
            "name": config.tts.provider,
            "healthy": tts_provider.is_healthy(),
            "voices": tts_provider.list_available_voices()[:5]
        },
        "avatar_provider": {
            "name": config.avatar.provider,
            "healthy": avatar_provider.is_healthy()
        },
        "security": {
            "local_only": config.security.local_only,
            "telemetry_disabled": config.security.disable_telemetry
        }
    }


@app.get("/api/eras")
async def get_eras():
    """Lists available chronological persona snapshots."""
    return [s.to_dict() for s in personality.list_snapshots()]


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Processes conversational turn with grounded retrieval, citations, optional TTS, and era-aware avatar rendering."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    resp = conversation_engine.process_turn(
        query=req.query,
        era_id=req.era_id,
        verification_filter=req.verification_filter,
        force_simulation=req.force_simulation,
        session_id=req.session_id
    )

    audio_url = None
    audio_path = None
    should_synthesize_audio = req.synthesize_audio or req.render_avatar or req.synthesize_avatar

    if should_synthesize_audio:
        try:
            filename = f"chat_{uuid.uuid4().hex[:10]}.wav"
            out_path = os.path.join(config.storage.audio_output_dir, filename)
            tts_provider.synthesize(
                text=resp.text,
                delivery=resp.recommended_delivery,
                output_file_path=out_path
            )
            audio_url = f"/api/audio/{filename}"
            audio_path = out_path
        except Exception as e:
            print(f"[Chat] TTS error: {e}")

    avatar_job_dict = None
    if (req.render_avatar or req.synthesize_avatar) and audio_path and os.path.exists(audio_path):
        try:
            target_era = resp.era_id or req.era_id
            job = avatar_provider.create_render_job(
                audio_path=audio_path,
                options={"era_id": target_era}
            )
            avatar_job_dict = job.to_dict()
            if target_era:
                avatar_job_dict["era_id"] = target_era
        except Exception as e:
            print(f"[Chat] Avatar render error: {e}")

    result = resp.to_dict()
    result["audio_url"] = audio_url
    if avatar_job_dict:
        result["avatar_job"] = avatar_job_dict
        result["avatar_job_id"] = avatar_job_dict.get("job_id")
    return result


@app.get("/api/chat/sessions")
async def list_chat_sessions():
    """Lists all active and historical chat sessions."""
    return db.list_sessions()


@app.post("/api/chat/sessions")
async def create_chat_session(req: SessionCreateRequest = SessionCreateRequest()):
    """Creates a new conversational session."""
    sid = db.create_session(title=req.title or "New Conversation")
    return {"session_id": sid, "title": req.title}


@app.get("/api/chat/sessions/{session_id}")
async def get_chat_session_details(session_id: str):
    """Retrieves session metadata and conversation history."""
    sess = db.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = db.get_session_messages(session_id, limit=50)
    return {"session": sess, "messages": messages}



@app.post("/api/tts/synthesize")
async def synthesize_tts(req: TTSRequest):
    """Synthesizes text into 24kHz audio matching delivery parameters."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        emotion_cat = EmotionCategory(req.emotion)
    except ValueError:
        emotion_cat = EmotionCategory.NEUTRAL

    delivery = ExpressiveDeliveryParams(
        emotion=emotion_cat,
        intensity=req.intensity,
        speech_rate=req.speech_rate,
        pitch_shift=req.pitch_shift
    )

    filename = f"tts_{uuid.uuid4().hex[:10]}.wav"
    out_path = os.path.join(config.storage.audio_output_dir, filename)

    try:
        tts_provider.synthesize(
            text=req.text,
            delivery=delivery,
            output_file_path=out_path,
            voice_override=req.voice
        )
        return {
            "status": "success",
            "audio_url": f"/api/audio/{filename}",
            "filename": filename,
            "delivery": delivery.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audio/{filename}")
async def serve_audio(filename: str):
    """Serves synthesized audio files from local cache."""
    file_path = Path(config.storage.audio_output_dir) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(file_path), media_type="audio/wav")


@app.post("/api/asr/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribes user speech from push-to-talk microphone audio (webm/wav/mp4) via local Whisper."""
    import tempfile
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        content = await audio.read()
        tmp.write(content)

    try:
        segments = asr_provider.transcribe(tmp_path, source_id="user_mic")
        full_text = " ".join(seg.text for seg in segments).strip()
        total_duration = segments[-1].end_seconds if segments else 0.0
        return {
            "status": "ok",
            "transcription": full_text,
            "segments": [s.to_dict() for s in segments],
            "duration_seconds": total_duration,
            "language": segments[0].language if segments else "en"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR transcription error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.get("/api/memories")
async def list_memories(limit: int = 100, status: Optional[str] = None):
    """Lists memories from SQLite archive with optional verification filter."""
    mems = db.list_memories(limit=limit, status_filter=status)
    return [m.to_dict() for m in mems]


@app.post("/api/memories/correct")
async def correct_memory(req: MemoryCorrectionRequest):
    """Applies a user correction to a memory and promotes it to personal_verified."""
    mem = db.get_memory(req.memory_id)
    if not mem:
        raise HTTPException(status_code=404, detail="Memory not found")

    corr = MemoryCorrection(
        memory_id=req.memory_id,
        original_content=mem.content,
        corrected_content=req.corrected_content,
        correction_reason=req.correction_reason,
        corrected_by="user",
        created_at=datetime.utcnow().isoformat()
    )

    success = db.correct_memory(corr)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to apply memory correction")

    # Update vector embedding
    updated_mem = db.get_memory(req.memory_id)
    if updated_mem:
        vector_store.add_memory_embedding(updated_mem)

    return {"status": "success", "memory": updated_mem.to_dict() if updated_mem else None}


@app.get("/api/beliefs")
async def list_beliefs(topic: Optional[str] = None):
    """Lists temporal beliefs illustrating value evolution across life eras."""
    if topic:
        beliefs = db.get_beliefs_for_topic(topic)
    else:
        beliefs = db.list_all_beliefs()
    return [b.to_dict() for b in beliefs]


@app.get("/api/archive/stats")
async def get_stats():
    """Archive counts and storage overview."""
    stats = db.get_archive_stats()
    stats["models"] = llm.get_model_info()
    return stats


@app.get("/api/media")
async def list_media():
    """Lists all ingested media source files."""
    media_list = db.list_all_media()
    return [m.to_dict() for m in media_list]


@app.get("/api/media/{source_id}/clip")
async def get_media_clip(source_id: str, start: float, end: float):
    """
    Streams a sliced WAV audio clip on-the-fly from the local audio cache.
    Works 100% locally from the internal APFS cache with /Volumes/SSD unmounted.
    """
    # Bounds validation
    if start < 0:
        raise HTTPException(status_code=400, detail="Start timestamp must be non-negative (start >= 0).")
    if end <= start:
        raise HTTPException(status_code=400, detail="End timestamp must be strictly greater than start (start < end).")

    # 1. Resolve local audio file path from cache or database metadata
    audio_path = Path(config.storage.audio_output_dir) / f"{source_id}.wav"
    if not audio_path.exists():
        audio_path = Path(config.storage.data_dir) / "audio_cache" / f"{source_id}.wav"

    if not audio_path.exists():
        media_src = db.get_media_by_id(source_id)
        if media_src and media_src.metadata:
            local_path = media_src.metadata.get("local_audio_path")
            if local_path and os.path.exists(local_path):
                audio_path = Path(local_path)

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio cache file for source_id '{source_id}' not found.")

    # 2. Slice audio segment using soundfile
    try:
        import soundfile as sf
        import io

        with sf.SoundFile(str(audio_path)) as sf_file:
            sr = sf_file.samplerate
            total_frames = len(sf_file)
            duration = total_frames / sr

            if start >= duration:
                raise HTTPException(
                    status_code=400,
                    detail=f"Start timestamp ({start:.2f}s) exceeds audio duration ({duration:.2f}s)."
                )

            if end > duration + 1.0:
                raise HTTPException(
                    status_code=400,
                    detail=f"End timestamp ({end:.2f}s) exceeds audio duration ({duration:.2f}s)."
                )
            clamped_end = min(end, duration)

            start_frame = max(0, int(start * sr))
            num_frames = max(1, int((clamped_end - start) * sr))

            sf_file.seek(start_frame)
            slice_data = sf_file.read(num_frames)

            buf = io.BytesIO()
            sf.write(buf, slice_data, sr, format="WAV")
            wav_bytes = buf.getvalue()

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'inline; filename="{source_id}_{start:.1f}_{end:.1f}.wav"',
                "Content-Length": str(len(wav_bytes)),
                "Accept-Ranges": "bytes"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error slicing audio clip: {str(e)}")


@app.post("/api/ingestion/run")
async def run_ingestion(req: IngestionRequest):
    """Triggers ingestion pipeline for a target folder or synthetic fixture."""
    res = ingestion_pipeline.run_ingestion_job(target_directory=req.target_directory)
    return res


@app.post("/api/avatar/render")
async def create_avatar_render(
    request: Request,
    audio_filename: Optional[str] = Form(None),
    reference_portrait: Optional[str] = Form(None),
    era_id: Optional[str] = Form(None),
    mode: Optional[str] = Form(None)
):
    """Enqueues an audio-driven avatar render job with synchronized H.264+AAC muxing and era-awareness."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                audio_filename = body.get("audio_filename") or audio_filename
                reference_portrait = body.get("reference_portrait") or reference_portrait
                era_id = body.get("era_id") or era_id
                mode = body.get("mode") or mode
        except Exception:
            pass

    if not audio_filename:
        raise HTTPException(status_code=400, detail="audio_filename is required")

    candidate_paths = [
        os.path.join(config.storage.audio_output_dir, audio_filename),
        os.path.join(config.storage.data_dir, "audio_cache", audio_filename),
        os.path.join(config.storage.data_dir, "audio_output", audio_filename),
        audio_filename
    ]
    audio_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if not audio_path:
        raise HTTPException(status_code=404, detail="Source audio file not found")

    options: Dict[str, Any] = {}
    if era_id:
        options["era_id"] = era_id
    if mode:
        options["mode"] = mode

    job = avatar_provider.create_render_job(
        audio_path=audio_path,
        reference_portrait=reference_portrait,
        options=options,
        era_id=era_id
    )
    res = job.to_dict()
    if era_id:
        res["era_id"] = era_id
    return res


@app.get("/api/avatar/status/{job_id}")
async def get_avatar_status(job_id: str):
    """Queries avatar render job status and output video path."""
    job = avatar_provider.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    res = job.to_dict()
    if hasattr(job, "era_id") and job.era_id:
        res["era_id"] = job.era_id
    return res


@app.get("/api/avatar/video/{filename}")
async def serve_avatar_video(filename: str):
    """Serves rendered avatar MP4 video."""
    video_path = Path(config.storage.data_dir) / "avatar_cache" / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(str(video_path), media_type="video/mp4")


@app.get("/api/avatar/portrait")
async def serve_avatar_portrait(era_id: Optional[str] = None):
    """Serves the canonical reference portrait image, optionally for a specific era."""
    portrait_path = avatar_provider.resolve_reference_portrait(era_id=era_id)
    if not os.path.exists(portrait_path):
        raise HTTPException(status_code=404, detail="Reference portrait not found")
    return FileResponse(portrait_path, media_type="image/png")


@app.post("/api/backup/create")
async def create_backup():
    """Creates a timestamped .tar.gz archive backup with SHA-256 manifest."""
    res = backup_mgr.create_full_backup()
    return res


@app.get("/api/config")
async def get_config():
    """Returns application configuration."""
    return config.to_dict()
