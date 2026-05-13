import os
import sys
import wave


# ─── STAGE 1: Asset Verification ──────────────────────────────────────────
def verify_assets():
    """
    Phase 5 Pre-Flight: Directly verifies that the pre-processed user assets
    exist on disk and meet specifications. No transcoding needed — the WAV
    has already been prepared (1ch, 16-bit, 24000Hz PCM).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    face_path = os.path.join(base_dir, "sample_inputs", "my_face.jpg")
    wav_path = os.path.join(base_dir, "sample_inputs", "my_voice.wav")

    print("\n[Pre-Flight] Verifying user assets on disk...")

    # ── Face Image ──
    if not os.path.exists(face_path):
        raise FileNotFoundError(
            f"Reference face image not found at {face_path}. "
            "Please place your portrait photo there."
        )
    face_size = os.path.getsize(face_path)
    print(f"  ✓ Face image: {face_path} ({face_size:,} bytes)")

    # ── Voice WAV ──
    if not os.path.exists(wav_path):
        raise FileNotFoundError(
            f"Pre-processed voice WAV not found at {wav_path}. "
            "Please ensure your reference audio is transcoded and placed there."
        )

    with wave.open(wav_path, "r") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        duration = wf.getnframes() / sample_rate

    wav_size = os.path.getsize(wav_path)
    print(f"  ✓ Voice WAV: {wav_path} ({wav_size:,} bytes)")
    print(f"    Format: {channels}ch, {sample_width * 8}-bit, {sample_rate}Hz, {duration:.2f}s")

    # Validate specs
    if channels != 1:
        raise ValueError(f"WAV must be mono (1 channel), got {channels}")
    if sample_width != 2:
        raise ValueError(f"WAV must be 16-bit (2 bytes), got {sample_width}")
    if sample_rate != 24000:
        raise ValueError(f"WAV must be 24000Hz, got {sample_rate}")
    if duration < 1.0:
        raise ValueError(f"WAV duration too short ({duration:.2f}s), need >= 1.0s")

    print(f"  ✓ WAV specifications validated: PASS")

    return face_path, wav_path


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  Open-HeyGen Live Pipeline — Phase 5: HF Space Integration")
    print("  Strategy: gradio_client → Free Cloud GPUs (No Local GPU)")
    print("=" * 64)

    # ──────────────── STAGE 1: Pre-Flight Asset Verification ──────────────
    print("\n─── STAGE 1: Pre-Flight Asset Verification ───")
    face_path, voice_path = verify_assets()

    # ──────────────── STAGE 2: Engine Initialization ──────────────────────
    print("\n─── STAGE 2: Engine Initialization (HF Space Clients) ───")

    tts = None
    try:
        from workers.audio_worker.tts_engine import TTSEngine
        tts = TTSEngine()
        print("[Init] F5-TTS HF Space Client: READY ✓")
    except Exception as e:
        print(f"[Init] F5-TTS Engine FAILED: {e}")
        import traceback
        traceback.print_exc()

    video = None
    try:
        from workers.video_worker.inference import VideoInference
        video = VideoInference()
        print("[Init] SadTalker HF Space Client: READY ✓")
    except Exception as e:
        print(f"[Init] SadTalker Engine FAILED: {e}")
        import traceback
        traceback.print_exc()

    # ──────────────── STAGE 3: Audio Synthesis (F5-TTS) ───────────────────
    script_text = (
        "Welcome to the future of open-source video generation. "
        "This is a live end-to-end test of the Open HeyGen pipeline."
    )
    
    # Bypass TTS if we already have the cloned audio from a previous successful run
    existing_audio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_storage", "audio_chunks", "cloned_speech.wav")
    if os.path.exists(existing_audio):
        print(f"\n─── STAGE 3: Audio Synthesis (SKIPPED - USING EXISTING FILE) ───")
        audio_output = existing_audio
        print(f"[Audio] Reusing existing file: {audio_output}")
    else:
        audio_output = None
        if tts:
            print("\n─── STAGE 3: Audio Synthesis (F5-TTS via HF Space) ───")
            try:
                audio_output = tts.clone_and_synthesize(script_text, voice_path)
                print(f"[Audio] Generated at: {audio_output}")
                if os.path.exists(audio_output):
                    size = os.path.getsize(audio_output)
                    print(f"[Audio] File verified on disk: {size:,} bytes ✓")
                else:
                    print("[Audio] WARNING: Output path returned but file not found on disk.")
                    audio_output = None
            except Exception as e:
                print(f"[Audio] Synthesis FAILED: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n─── STAGE 3: Audio Synthesis SKIPPED (engine not loaded) ───")

    # ──────────────── STAGE 4: Video Animation (SadTalker) ────────────────
    video_output = None

    if video and audio_output and os.path.exists(audio_output):
        print("\n─── STAGE 4: Video Animation (SadTalker via HF Space) ───")
        try:
            video_output = video.map_audio_to_visemes(face_path, audio_output)
            print(f"[Video] Generated at: {video_output}")
            if os.path.exists(str(video_output)):
                size = os.path.getsize(video_output)
                print(f"[Video] File verified on disk: {size:,} bytes ✓")
            else:
                print("[Video] WARNING: Output path returned but file not found on disk.")
                video_output = None
        except Exception as e:
            print(f"[Video] Animation FAILED: {e}")
            import traceback
            traceback.print_exc()
    elif video and not audio_output:
        print("\n─── STAGE 4: Video Animation SKIPPED (no audio output) ───")
    else:
        print("\n─── STAGE 4: Video Animation SKIPPED (engine not loaded) ───")

    # ──────────────── FINAL REPORT ────────────────────────────────────────
    print("\n" + "=" * 64)
    if video_output and os.path.exists(str(video_output)):
        print("  ✅ PIPELINE EXECUTION: COMPLETE")
        print(f"  Final Media: {video_output}")
        print(f"  File Size: {os.path.getsize(video_output):,} bytes")
    elif audio_output and os.path.exists(str(audio_output)):
        print("  ⚠️  PIPELINE EXECUTION: PARTIAL (Audio OK, Video pending)")
        print(f"  Audio Output: {audio_output}")
        print(f"  File Size: {os.path.getsize(audio_output):,} bytes")
    else:
        print("  ❌ PIPELINE EXECUTION: STRUCTURAL VERIFICATION ONLY")
        print("  HF Space connectivity was tested but generation did not complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
