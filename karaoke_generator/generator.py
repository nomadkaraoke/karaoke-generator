import os
import hashlib
import datetime
import shutil
import yt_dlp
from audio_separator import Separator
from lyrics_transcriber import LyricsTranscriber


class KaraokeGenerator:
    def __init__(
        self,
        youtube_url=None,
        audio_file=None,
        output_dir=None,
        model_name="UVR_MDXNET_KARA_2",
        model_file_dir="/tmp/audio-separator-models",
        cache_dir="/tmp/karaoke-generator-cache",
        genius_api_token=False,
        song_artist=None,
        song_title=None,
    ):
        log("KaraokeGenerator initializing")

        self.model_name = model_name
        self.model_file_dir = model_file_dir
        self.cache_dir = cache_dir
        self.output_dir = output_dir

        self.genius_api_token=genius_api_token,
        self.song_artist=song_artist,
        self.song_title=song_title,
        
        if audio_file is None and youtube_url is None:
            raise Exception(
                "Either audio_file or youtube_url must be specified as the input source"
            )

        self.youtube_url = youtube_url
        self.audio_file = audio_file

        self.create_folders()

    def generate(self):
        log("KaraokeGenerator beginning generation")

        if self.audio_file is None and self.youtube_url is not None:
            log(
                f"audio_file is none and youtube_url is {self.youtube_url}, fetching video with yt-dlp"
            )

            ydl_opts = {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                    }
                ],
                "outtmpl": os.path.join(self.cache_dir, "%(id)s_%(title)s.%(ext)s")
            }

            # Download and convert the audio to WAV
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.youtube_url, download=False)
                orig_filepath = ydl.prepare_filename(info)

                wav_filepath = os.path.splitext(orig_filepath)[0] + ".wav"
                log(f"wav_filepath: {wav_filepath}")

                if os.path.isfile(wav_filepath):
                    log(f"found existing file at wav_filepath, skipping download: {wav_filepath}")
                else:
                    ydl.download([self.youtube_url])

                if self.output_dir:
                    shutil.copy(wav_filepath, self.output_dir)

                # Update self.audio_file with the new file path
                self.audio_file = wav_filepath
                log(f"audio_file updated: {self.audio_file}")

        elif self.audio_file is None:
            raise Exception("No audio source provided.")

        log(f"audio_file is set: {self.audio_file}, instantiating Separator")

        separator = Separator(
            self.audio_file,
            model_name=self.model_name,
            model_file_dir=self.model_file_dir,
            output_dir=self.output_dir,
        )
        primary_stem_path, secondary_stem_path = separator.separate()

        print(
            f"Separation complete! Output files: {primary_stem_path} {secondary_stem_path}"
        )

        transcriber = LyricsTranscriber(
            self.audio_file,
            genius_api_token=self.genius_api_token,
            song_artist=self.song_artist,
            song_title=self.song_title,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
        )

        log("LyricsTranscriber beginning transcription")

        result_metadata = transcriber.generate()

        log(f"*** Success! ***")

        formatted_duration = f'{int(result_metadata["song_duration"] // 60):02d}:{int(result_metadata["song_duration"] % 60):02d}'
        log(f"Total Song Duration: {formatted_duration}")

        formatted_singing_duration = f'{int(result_metadata["total_singing_duration"] // 60):02d}:{int(result_metadata["total_singing_duration"] % 60):02d}'
        log(f"Total Singing Duration: {formatted_singing_duration}")
        log(f"Singing Percentage: {result_metadata['singing_percentage']}%")

        log(f"*** Outputs: ***")
        log(
            f"Whisper transcription output JSON file: {result_metadata['whisper_json_filepath']}"
        )
        log(f"MidiCo LRC output file: {result_metadata['midico_lrc_filepath']}")
        log(f"Genius lyrics output file: {result_metadata['genius_lyrics_filepath']}")

    def create_folders(self):
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        if self.model_file_dir is not None:
            os.makedirs(self.model_file_dir, exist_ok=True)

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)


def log(message):
    timestamp = datetime.datetime.now().isoformat()
    print(f"{timestamp} - {message}")
