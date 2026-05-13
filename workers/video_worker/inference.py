import os
import shutil
import time
import concurrent.futures
from gradio_client import Client, handle_file

class VideoInference:
    def __init__(self):
        self.space_id = "fffiloni/LivePortrait"
        print(f"[Video Engine] Connecting to accelerated cloud backend: {self.space_id}")
        
        def init_client():
            return Client(self.space_id, verbose=True)
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(init_client)
            try:
                self.client = future.result(timeout=10)
                print(f"[Video Engine] Connected to {self.space_id} ✓")
            except concurrent.futures.TimeoutError:
                raise ValueError(f"Timeout: {self.space_id} failed to initialize within 10 seconds.")
            except Exception as e:
                raise ValueError(f"Error initializing {self.space_id}: {e}")

    def map_audio_to_visemes(self, face_image_path: str, audio_track_path: str) -> str:
        """
        Sends the verified local user assets to the cloud GPU space.
        Returns the path to the downloaded output .mp4 file.
        """
        print(f"[Video Engine] Sending processing request to {self.space_id} API...")
        
        try:
            # We pass assets using the mandatory handle_file wrapper for Gradio v4+ stability
            # Also use /predict as the typical default if /generate_video isn't exactly the API name
            # Some spaces use /predict by default. If we can guess the API endpoint, we can also try /process or similar
            result = self.client.predict(
                img_input=handle_file(face_image_path),
                audio_input=handle_file(audio_track_path),
                api_name="/generate_video"
            )
        except Exception as e:
            print(f"[Video Engine] /generate_video endpoint failed, attempting default /predict. Error: {e}")
            try:
                result = self.client.predict(
                    handle_file(face_image_path),
                    handle_file(audio_track_path),
                    api_name="/predict"
                )
            except Exception as e2:
                raise RuntimeError(f"All API endpoints failed on {self.space_id}: {e2}")
        
        if isinstance(result, (list, tuple)):
            res = result[0]
            if isinstance(res, dict):
                result = res.get("video") or res.get("name") or res
            else:
                result = res
        elif isinstance(result, dict):
            result = result.get("video") or result.get("name") or result
            
        # Save a copy out of temp folders into our local project architecture
        # We need to make sure the output directory is mock_storage/output_videos as per the original architecture
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "mock_storage", "output_videos")
        os.makedirs(output_dir, exist_ok=True)
        
        local_output_path = os.path.join(output_dir, f"live_generation.mp4")
        if result and os.path.exists(str(result)):
            shutil.copy(str(result), local_output_path)
        else:
            raise RuntimeError(f"Video file not generated successfully. Result: {result}")
        
        return local_output_path
