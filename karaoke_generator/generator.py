import os
import shutil
import logging
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
        log_format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
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

        self.logger.debug(f"audio_file is valid file: {self.audio_file}, instantiating Separator")

        separator = Separator(
            self.audio_file,
            model_name=self.model_name,
            model_file_dir=self.model_file_dir,
            output_dir=self.cache_dir,
        )
        self.primary_stem_path, self.secondary_stem_path = separator.separate()

        self.logger.debug(f"Separation complete! Output files: {self.primary_stem_path} {self.secondary_stem_path}")

        self.output_values["primary_stem_path"] = self.primary_stem_path
        self.output_values["secondary_stem_path"] = self.secondary_stem_path

        if self.output_dir:
            shutil.copy(self.primary_stem_path, self.output_dir)
            shutil.copy(self.secondary_stem_path, self.output_dir)

    def download_youtube_video(self):
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
            temp_download_parts = os.path.splitext(temp_download_filepath)
            self.logger.debug(f"temp_download_filepath: {temp_download_filepath}")

            # Create a slugified download filepath to get rid of spaces and unicode characters from youtube titles,
            # but retain original file extension (which may vary depending on the video format youtube returns)
            temp_download_filename_parts = os.path.splitext(os.path.basename(temp_download_filepath))
            filename_slug = slugify.slugify(temp_download_filename_parts[0])

            download_filepath = os.path.join(self.cache_dir, filename_slug + temp_download_parts[1])
            self.logger.debug(f"download_filepath: {download_filepath}")

            if os.path.isfile(download_filepath):
                self.logger.debug(f"found existing file at download_filepath, skipping download: {download_filepath}")
            else:
                ydl.download([self.youtube_url])
                shutil.move(temp_download_filepath, download_filepath)
                self.youtube_video_file = download_filepath
                self.logger.debug(f"successfully downloaded youtube video to path: {self.youtube_video_file}")

        self.audio_file = os.path.join(self.cache_dir, filename_slug + ".wav")
        # TODO: extract audio from the downloaded video in WAV format using ffmpeg, writing to the filepath self.audio_file
        self.logger.debug(f"self.audio_file updated: {self.audio_file}")

        self.youtube_video_image_file = os.path.join(self.cache_dir, filename_slug + ".png")
        # TODO: extract a still image from 30 seconds into the downloaded video in PNG format using ffmpeg, writing to image_filepath
        self.logger.debug(f"self.youtube_video_image_file updated: {self.youtube_video_image_file}")

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
