import datetime
import os
import requests
import time
import torch
from cog import BasePredictor, BaseModel, Input, File, Path
from pyannote.audio import Pipeline
import subprocess


class Output(BaseModel):
    segments: list


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.diarization_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token="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").to(torch.device("cuda"))

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
                default=True),

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
            segments = self.diarize(temp_wav_filename)

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


    def diarize(self,
                wav_filename):
        time_start = time.time()

        diarization = self.diarization_model(wav_filename)
        time_diraization_end = time.time()
        diarization_list = list(diarization.itertracks(yield_label=True))

        print(
            f"Finished with diarization, took {time_diraization_end - time_start:.5} seconds"
        )
        return diarization_list
