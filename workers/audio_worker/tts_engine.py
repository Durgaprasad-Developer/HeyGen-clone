import os
import shutil
import time
import concurrent.futures
from gradio_client import Client, handle_file

class TTSEngine:
    def __init__(self):
        self.space_id = "SWivid/F5-TTS"
        print(f"[TTSEngine] Connecting to {self.space_id}...")
        
        def init_client():
            return Client(self.space_id, verbose=True)
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(init_client)
            try:
                self.client = future.result(timeout=10)
                print(f"[TTSEngine] Connected to {self.space_id} ✓")
            except concurrent.futures.TimeoutError:
                raise ValueError(f"Timeout: {self.space_id} failed to initialize within 10 seconds.")
            except Exception as e:
                raise ValueError(f"Error initializing {self.space_id}: {e}")

    def clone_and_synthesize(self, text: str, ref_audio_path: str, ref_text: str = "") -> str:
        if not text or text.strip() == "":
            raise ValueError("Script text empty.")
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(ref_audio_path)

        output_dir = "mock_storage/audio_chunks"
        os.makedirs(output_dir, exist_ok=True)

        print(f"[TTSEngine] Requesting synthesis from {self.space_id}...")
        start_time = time.time()
        
        try:
            # Common F5-TTS signature
            result = self.client.predict(
                ref_audio_input=handle_file(ref_audio_path),
                ref_text_input=ref_text,
                gen_text_input=text,
                remove_silence=False,
                api_name="/infer"
            )
        except Exception as e1:
            try:
                # Alternative signature
                result = self.client.predict(
                    ref_audio_orig=handle_file(ref_audio_path),
                    ref_text=ref_text,
                    gen_text=text,
                    exp_name="F5-TTS",
                    remove_silence=False,
                    cross_fade_duration=0.15,
                    api_name="/infer"
                )
            except Exception as e2:
                try:
                    result = self.client.predict(
                        ref_audio=handle_file(ref_audio_path),
                        ref_text=ref_text,
                        gen_text=text,
                        remove_silence=False,
                        api_name="/predict"
                    )
                except Exception as e3:
                    raise RuntimeError(f"All TTS API calls failed. E1: {e1}, E2: {e2}, E3: {e3}")

        if isinstance(result, (list, tuple)):
            result_path = result[0]
        else:
            result_path = result

        elapsed = time.time() - start_time
        print(f"[TTSEngine] Completed in {elapsed:.1f}s")

        if result_path and os.path.exists(str(result_path)):
            output_path = os.path.join(output_dir, "cloned_speech.wav")
            shutil.copy2(str(result_path), output_path)
            return output_path
        else:
            raise RuntimeError(f"No result file from API: {result}")
