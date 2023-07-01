import os
import hashlib
import datetime
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
        model_file_dir="/tmp/audio-separator-models/",
        cache_dir="/tmp/karaoke-generator-cache/",
    ):
        log("KaraokeGenerator initializing")

        self.model_name = model_name
        self.model_file_dir = model_file_dir
        self.cache_dir = cache_dir
        self.output_dir = output_dir

        if audio_file is None and youtube_url is None:
            raise Exception(
                "Either audio_file or youtube_url must be specified as the input source"
            )

        self.youtube_url = youtube_url
        self.audio_file = audio_file

        self.create_folders()

    def get_file_hash(self, filepath):
        return hashlib.md5(open(filepath, "rb").read()).hexdigest()

    def generate(self):
        log("KaraokeGenerator beginning generation")

        if self.audio_file is None and self.youtube_url is not None:
            log(
                f"audio_file is none and youtube_url is {self.youtube_url}, fetching video with yt-dlp"
            )

            URLS = [self.youtube_url]

            ydl_opts = {
                "format": "bestaudio/best",
                "postprocessors": [
                    {  # Extract audio using ffmpeg
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                    }
                ],
            }

            # Download and convert the audio to WAV
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.youtube_url])

            # Calculate the hash of the file contents
            temp_filepath = f"{self.cache_dir}/temp.wav"
            file_hash = self.get_file_hash(temp_filepath)

            # Rename the file to its hash and move it to the cache directory
            final_filepath = f"{self.cache_dir}/{file_hash}.wav"
            os.rename(temp_filepath, final_filepath)

            # Update self.audio_file with the new file path
            self.audio_file = final_filepath
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

        formatted_singing_duration = (
            f'{int(result_metadata["total_singing_duration"] // 60):02d}:{int(result_metadata["total_singing_duration"] % 60):02d}'
        )
        log(f"Total Singing Duration: {formatted_singing_duration}")
        log(f"Singing Percentage: {result_metadata['singing_percentage']}%")

        log(f"*** Outputs: ***")
        log(f"Whisper transcription output JSON file: {result_metadata['whisper_json_filepath']}")
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
