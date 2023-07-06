#!/usr/bin/env python
import argparse
import logging
from karaoke_generator import KaraokeGenerator


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(message)s")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    logger.debug("Parsing CLI args")

    parser = argparse.ArgumentParser(description="Generate karaoke music video for either a local audio file or YouTube URL")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--youtube_url",
        default=None,
        help="Optional: YouTube URL to make karaoke version of.",
    )
    input_group.add_argument(
        "--audio_file",
        default=None,
        help="Optional: audio file path to make karaoke version of.",
    )

    parser.add_argument(
        "--song_artist",
        default=None,
        help="Optional: specify song artist for Genius lyrics lookup and auto-correction",
    )
    parser.add_argument(
        "--song_title",
        default=None,
        help="Optional: specify song title for Genius lyrics lookup and auto-correction",
    )

    parser.add_argument(
        "--genius_api_token",
        default=None,
        help="Optional: specify Genius API token for lyrics lookup and auto-correction",
    )

    parser.add_argument(
        "--model_name",
        default="UVR_MDXNET_KARA_2",
        help="Optional: model name to be used for audio separation.",
    )
    parser.add_argument(
        "--model_file_dir",
        default="/tmp/audio-separator-models/",
        help="Optional: audio separation model files directory.",
    )
    parser.add_argument(
        "--cache_dir",
        default="/tmp/karaoke-generator-cache/",
        help="Optional: directory to cache generated files to avoid wasted computation.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional: directory to write output files to. Defaults to a folder in the current directory.",
    )

    args = parser.parse_args()

    logger.info(f"Karaoke generator beginning with audio_file: {args.audio_file} / youtube_url: {args.youtube_url}")

    generator = KaraokeGenerator(
        audio_file=args.audio_file,
        youtube_url=args.youtube_url,
        song_artist=args.song_artist,
        song_title=args.song_title,
        genius_api_token=args.genius_api_token,
        model_name=args.model_name,
        model_file_dir=args.model_file_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    output_files = generator.generate()

    logger.info(f"Karaoke generation complete! Outputs: ")
    for key in output_files:
        if type(output_files[key]) is dict:
            for key2 in output_files[key]:
                print(f"{key} / {key2}: {output_files[key][key2]}")
        else:
            print(f"{key}: {output_files[key]}")


if __name__ == "__main__":
    main()
