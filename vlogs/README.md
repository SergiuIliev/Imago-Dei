# Personal Vlog & Media Recordings Directory

This is the dedicated folder for your personal recordings in the **Personal AI Time Capsule** (`/Users/Sergiu/AI/Time_Capsule/vlogs`).

## How to use:
1. Copy or move your personal video or audio recordings into this folder:
   - Supported video formats: `.mp4`, `.mov`, `.mkv`
   - Supported audio formats: `.wav`, `.m4a`, `.mp3`, `.flac`
2. Open the Time Capsule Dashboard (`http://127.0.0.1:7860`) and navigate to the **Archive & Media** tab.
3. The pipeline will automatically scan this folder, compute cryptographic SHA-256 deduplication hashes, run Whisper ASR with word-level timestamps, extract atomic memories with provenance metadata, and index them into the memory vault.
4. Your source recordings remain strictly on your local disk at all times. Zero data is ever uploaded or transmitted externally.
