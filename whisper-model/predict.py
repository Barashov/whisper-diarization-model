# Prediction interface for Cog ⚙️

import base64
import datetime
import os
import requests
import time
import torch
import subprocess
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

    def predict(self,
                file_path: str = Input(description='file path in docker volume',
                                       default=''),

                file_url: str = Input(description="Or provide: A direct audio file URL",
                                      default=''),

                prompt: str = Input(description="Prompt, to be used as context",
                                    default="Some people speaking."),
                offset_seconds: int = Input(description="Offset in seconds, used for chunked inputs",
                                            default=0,
                                            ge=0)) -> Output:

        temp_wav_filename = f"temp-{time.time_ns()}.wav"
        try:
            if file_url:
                response = requests.get(file_url)

                with open(temp_wav_filename, 'wb') as file:
                    file.write(response.content)
            else:
                temp_wav_filename = file_path

            segments = self.speech_to_text(temp_wav_filename,
                                           prompt=prompt,
                                           offset_seconds=offset_seconds)

            return Output(segments=segments)

        except Exception as e:
            print(e)

        finally:
            # Clean up
            if os.path.exists(temp_wav_filename):
                os.remove(temp_wav_filename)

    def convert_time(self, secs, offset_seconds=0):
        return datetime.timedelta(seconds=(round(secs) + offset_seconds))

    def speech_to_text(self,
                       audio_file_wav,
                       prompt="People talking.",
                       offset_seconds=0):
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
