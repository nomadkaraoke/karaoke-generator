import os
import json
import urllib
import shutil
import logging
import subprocess
import yt_dlp
import slugify
from audio_separator import Separator
from lyrics_transcriber import LyricsTranscriber


class KaraokeGenerator:
    def __init__(
        self,
        youtube_url=None,
        audio_file=None,
        song_artist=None,
        song_title=None,
        genius_api_token=None,
        model_name="UVR_MDXNET_KARA_2",
        model_file_dir="/tmp/audio-separator-models",
        cache_dir="/tmp/karaoke-generator-cache",
        output_dir=None,
        log_level=logging.DEBUG,
        log_format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        log_handler = logging.StreamHandler()
        log_formatter = logging.Formatter(log_format)
        log_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_handler)

        self.logger.debug("KaraokeGenerator initializing")

        self.model_name = model_name
        self.model_file_dir = model_file_dir
        self.cache_dir = cache_dir
        self.output_dir = output_dir

        self.genius_api_token = genius_api_token
        self.song_artist = song_artist
        self.song_title = song_title

        if audio_file is None and youtube_url is None:
            raise Exception("Either audio_file or youtube_url must be specified as the input source")
        if audio_file is not None and youtube_url is not None:
            raise Exception("Only one of audio_file or youtube_url may be specified as the input source")

        if audio_file is not None:
            self.input_source_slug = slugify.slugify(os.path.basename(audio_file), lowercase=False)
        if youtube_url is not None:
            parsed_url = urllib.parse.urlparse(youtube_url)
            self.input_source_slug = slugify.slugify(parsed_url.hostname + "-" + parsed_url.query, lowercase=False)

        if self.output_dir is None:
            self.output_dir = os.path.join(os.getcwd(), "karaoke-generator-output-" + self.input_source_slug)

        self.output_filename_slug = None
        self.youtube_url = youtube_url
        self.youtube_video_file = None
        self.youtube_video_image_file = None

        self.audio_file = audio_file
        self.primary_stem_path = None
        self.secondary_stem_path = None

        self.output_values = {}
        self.create_folders()

    def generate(self):
        self.logger.info("KaraokeGenerator beginning generation")

        if self.audio_file is None and self.youtube_url is not None:
            self.logger.debug(f"audio_file is none and youtube_url is {self.youtube_url}, fetching video from youtube")
            self.download_youtube_video()

        self.separate_audio()
        self.transcribe_lyrics()

        self.logger.info("KaraokeGenerator complete!")

        return self.output_values

    def transcribe_lyrics(self):
        transcriber = LyricsTranscriber(
            self.audio_file,
            genius_api_token=self.genius_api_token,
            song_artist=self.song_artist,
            song_title=self.song_title,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
        )

        transcription_metadata = transcriber.generate()

        self.logger.debug(f"Transcription Success!")

        formatted_duration = (
            f'{int(transcription_metadata["song_duration"] // 60):02d}:{int(transcription_metadata["song_duration"] % 60):02d}'
        )
        formatted_singing_duration = f'{int(transcription_metadata["total_singing_duration"] // 60):02d}:{int(transcription_metadata["total_singing_duration"] % 60):02d}'

        transcription_metadata["formatted_duration"] = formatted_duration
        transcription_metadata["formatted_singing_duration"] = formatted_singing_duration

        self.output_values["transcription_metadata"] = transcription_metadata

        self.logger.debug(f"Total Song Duration: {formatted_duration}")
        self.logger.debug(f"Total Singing Duration: {formatted_singing_duration}")
        self.logger.debug(f"Singing Percentage: {transcription_metadata['singing_percentage']}%")
        self.logger.debug(f"Whisper transcription output JSON file: {transcription_metadata['whisper_json_filepath']}")
        self.logger.debug(f"MidiCo LRC output file: {transcription_metadata['midico_lrc_filepath']}")
        self.logger.debug(f"Genius lyrics output file: {transcription_metadata['genius_lyrics_filepath']}")

        if self.output_dir:
            shutil.copy(self.output_values["transcription_metadata"]["midico_lrc_filepath"], self.output_dir)
            shutil.copy(self.output_values["transcription_metadata"]["genius_lyrics_filepath"], self.output_dir)

    def separate_audio(self):
        if self.audio_file is None or not os.path.isfile(self.audio_file):
            raise Exception("Error: Invalid audio source provided.")

        self.logger.debug(f"audio_file is valid file: {self.audio_file}")

        self.primary_stem_path = os.path.join(self.output_dir, f"{self.output_filename_slug}_(Instrumental)_{self.model_name}.wav")
        self.secondary_stem_path = os.path.join(self.output_dir, f"{self.output_filename_slug}_(Vocals)_{self.model_name}.wav")

        if os.path.isfile(self.primary_stem_path) and os.path.isfile(self.secondary_stem_path):
            self.logger.debug(f"Separated audio files already exist in output paths, skipping separation: {self.primary_stem_path}")
        else:
            self.logger.debug(f"instantiating Separator with model_name: {self.model_name} and output_dir: {self.output_dir}")
            separator = Separator(
                self.audio_file,
                model_name=self.model_name,
                model_file_dir=self.model_file_dir,
                output_dir=self.output_dir,
            )
            self.primary_stem_path, self.secondary_stem_path = separator.separate()

            self.logger.debug(f"Separation complete! Output files: {self.primary_stem_path} {self.secondary_stem_path}")

        self.output_values["primary_stem_path"] = self.primary_stem_path
        self.output_values["secondary_stem_path"] = self.secondary_stem_path

    def download_youtube_video(self):
        # Cache file path for the ydl.extract_info output
        ydl_info_cache_file = os.path.join(self.cache_dir, f"ydl-info-{self.input_source_slug}.json")

        # Check if the info cache file exists, if so, read from it
        if os.path.isfile(ydl_info_cache_file):
            self.logger.debug(f"Reading ydl info from cache: {ydl_info_cache_file}")
            with open(ydl_info_cache_file, "r") as cache_file:
                info = json.load(cache_file)
                self.youtube_video_file = info["download_filepath"]
                self.output_filename_slug = info["output_filename_slug"]
        else:
            self.logger.debug(f"No existing YDL info file found")

        if self.youtube_video_file is not None and os.path.isfile(self.youtube_video_file):
            self.logger.debug(f"Existing file found at self.youtube_video_file, skipping download: {self.youtube_video_file}")
        else:
            self.logger.debug(f"No existing file found, preparing download")
            # Options for downloading the original highest quality file to the cache dir
            ydl_opts = {
                "logger": YoutubeDLLogger(self.logger),
                "format": "best",
                "outtmpl": os.path.join(self.cache_dir, "%(id)s-%(title)s.%(ext)s"),
            }

            # Download the original highest quality file
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.youtube_url, download=False)

                temp_download_filepath = ydl.prepare_filename(info)
                self.logger.debug(f"temp_download_filepath: {temp_download_filepath}")

                download_file_extension = os.path.splitext(os.path.basename(temp_download_filepath))[1]
                self.logger.debug(f"download_file_extension: {download_file_extension}")

                # Create a slugified filename prefix to get rid of spaces and unicode characters from youtube titles
                info["output_filename_slug"] = info["id"] + "-" + slugify.slugify(info["title"], lowercase=False)
                self.output_filename_slug = info["output_filename_slug"]
                self.logger.debug(f"output_filename_slug: {self.output_filename_slug}")

                # but retain original file extension (which may vary depending on the video format youtube returns)
                info["download_filepath"] = os.path.join(self.cache_dir, self.output_filename_slug + download_file_extension)
                self.logger.debug(f"download_filepath: {info['download_filepath'] }")

                # Save the ydl.extract_info output to cache file
                self.logger.debug(f"Saving sanitized YT-DLP info to cache: {ydl_info_cache_file}")
                with open(ydl_info_cache_file, "w") as cache_file:
                    json.dump(ydl.sanitize_info(info), cache_file, indent=4)

                ydl.download([self.youtube_url])
                shutil.move(temp_download_filepath, info["download_filepath"])
                self.youtube_video_file = info["download_filepath"]
                self.logger.debug(f"successfully downloaded youtube video to path: {self.youtube_video_file}")

        # Extract audio to WAV file using ffmpeg
        self.audio_file = os.path.join(self.cache_dir, self.output_filename_slug + ".wav")
        if os.path.isfile(self.audio_file):
            self.logger.debug(f"Existing file found at self.audio_file, skipping extraction: {self.audio_file}")
        else:
            self.logger.debug(f"Extracting audio from {self.youtube_video_file} to {self.audio_file}")
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    self.youtube_video_file,  # Input file path
                    "-vn",  # Only keep the audio
                    "-f",
                    "wav",  # Output format
                    self.audio_file,  # Output file path
                ]
            )
            self.logger.debug(f"Audio extracted to self.audio_file filepath: {self.audio_file}")

        # Extract a still image from 30 seconds into the video using ffmpeg
        self.youtube_video_image_file = os.path.join(self.cache_dir, self.output_filename_slug + ".png")
        if os.path.isfile(self.youtube_video_image_file):
            self.logger.debug(f"Existing file found at self.youtube_video_image_file, skipping extraction: {self.youtube_video_image_file}")
        else:
            self.logger.debug(f"Extracting still image from {self.youtube_video_file} to {self.youtube_video_image_file }")
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    "30",  # Position to start extracting (30 seconds)
                    "-i",
                    self.youtube_video_file,  # Input file path
                    "-vframes",
                    "1",  # Only extract 1 frame (which will be a still image)
                    self.youtube_video_image_file,  # Output file path
                ]
            )
            self.logger.debug(f"Still image extracted to self.youtube_video_image_file filepath: {self.youtube_video_image_file}")

        self.output_values["youtube_video_file"] = self.youtube_video_file
        self.output_values["youtube_video_image_file"] = self.youtube_video_image_file
        self.output_values["audio_file"] = self.audio_file

        # Copy the files to an output directory if specified
        if self.output_dir:
            shutil.copy(self.youtube_video_file, self.output_dir)
            shutil.copy(self.youtube_video_image_file, self.output_dir)
            shutil.copy(self.audio_file, self.output_dir)

    def create_folders(self):
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        if self.model_file_dir is not None:
            os.makedirs(self.model_file_dir, exist_ok=True)

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.output_values["output_dir"] = self.output_dir


class YoutubeDLLogger:
    def __init__(self, logger):
        self.logger = logger

    def debug(self, msg):
        # For compatibility with youtube-dl, both debug and info are passed into debug
        # You can distinguish them by the prefix '[debug] '
        if msg.startswith("[debug] "):
            pass
        else:
            self.logger.info(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)
