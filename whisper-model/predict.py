# Prediction interface for Cog ⚙️

import base64
import datetime
import os
import requests
import time
import torch

from cog import BasePredictor, BaseModel, Input, File, Path
from faster_whisper import WhisperModel


class Output(BaseModel):
    segments: list


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model_name = "large-v2"
        self.model = WhisperModel(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16")

    def predict(
            self,
            file_path: str = Input(
                description='file path in docker volume',
                default=None),

            file_url: str = Input(
                description="Or provide: A direct audio file URL",
                default=None),

            use_ffmpeg: bool = Input(
                description="if true: convert file to wav. default true",
                default=True
            ),

            prompt: str = Input(description="Prompt, to be used as context",
                                default="Some people speaking."),
            offset_seconds: int = Input(
                description="Offset in seconds, used for chunked inputs",
                default=0,
                ge=0)
    ) -> Output:
        """Run a single prediction on the model"""
        # Check if either filestring, filepath or file is provided, but only 1 of them
        """ if sum([file_string is not None, file_url is not None, file is not None]) != 1:
            raise RuntimeError("Provide either file_string, file or file_url") """
        temp_wav_filename = f"temp-{time.time_ns()}.wav"
        try:
            if file_url:
                response = requests.get(file_url)
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                if use_ffmpeg:
                    with open(temp_audio_filename, 'wb') as file:
                        file.write(response.content)

                    subprocess.run([
                        'ffmpeg', '-i', temp_audio_filename, '-ar', '16000', '-ac',
                        '1', '-c:a', 'pcm_s16le', temp_wav_filename
                    ])
                else:
                    with open(temp_wav_filename, 'wb') as file:
                        file.write(response.content)

                if os.path.exists(temp_audio_filename):
                    os.remove(temp_audio_filename)
            else:
                temp_wav_filename = file_path
            segments = self.speech_to_text(temp_wav_filename,
                                           prompt=prompt,
                                           offset_seconds=offset_seconds)

            print(f'done with transcription')
            # Return the results as a JSON object
            return Output(segments=segments, error_text=None)

        except Exception as e:
            print(e)
            return Output(segments=None, error_text=e)

        finally:
            # Clean up
            if os.path.exists(temp_wav_filename):
                os.remove(temp_wav_filename)

    def convert_time(self, secs, offset_seconds=0):
        return datetime.timedelta(seconds=(round(secs) + offset_seconds))

    def speech_to_text(self,
                       audio_file_wav,
                       prompt="People talking.",
                       offset_seconds=0,
                       group_segments=True):
        time_start = time.time()

        # Transcribe audio
        print("Starting transcribing")
        options = dict(vad_filter=True,
                       initial_prompt=prompt,
                       word_timestamps=True)
        segments, _ = self.model.transcribe(audio_file_wav, **options)
        segments = list(segments)
        segments = [{
            'start':
                float(s.start + offset_seconds),
            'end':
                float(s.end + offset_seconds),
            'text':
                s.text,
            'words': [{
                'start': float(w.start + offset_seconds),
                'end': float(w.end + offset_seconds),
                'word': w.word
            } for w in s.words]
        } for s in segments]

        time_transcribing_end = time.time()
        print(
            f"Finished with transcribing, took {time_transcribing_end - time_start:.5} seconds"
        )

        return segments