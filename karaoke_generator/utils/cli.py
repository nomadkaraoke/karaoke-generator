#!/usr/bin/env python
import argparse
import datetime
from karaoke_generator import KaraokeGenerator


def log(message):
    timestamp = datetime.datetime.now().isoformat()
    print(f"{timestamp} - {message}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate karaoke music video for either a local audio file or YouTube URL"
    )

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

    args = parser.parse_args()

    log(
        f"Karaoke generator beginning with audio_file: {args.audio_file} / youtube_url: {args.youtube_url}"
    )

    generator = KaraokeGenerator(
        audio_file=args.audio_file,
        youtube_url=args.youtube_url,
        model_name=args.model_name,
        model_file_dir=args.model_file_dir,
        cache_dir=args.cache_dir,
    )
    output_dir_path, output_video_path = generator.generate()

    log(
        f"Karaoke generation complete! Output dir: {output_dir_path} Karaoke video: {output_video_path}"
    )


if __name__ == "__main__":
    main()
