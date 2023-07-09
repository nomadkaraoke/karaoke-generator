#!/usr/bin/env python
import argparse
import logging
import pkg_resources
from karaoke_generator import KaraokeGenerator


def main():
    logger = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    parser = argparse.ArgumentParser(
        description="Generate karaoke music video for either a local audio file or YouTube URL",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=40),
    )

    parser.add_argument(
        "input_path", nargs="?", help="The audio file path or YouTube URL to make karaoke version of.", default=argparse.SUPPRESS
    )

    package_version = pkg_resources.get_distribution("karaoke-generator").version
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {package_version}")
    parser.add_argument("--log_level", default="INFO", help="Optional: Logging level, e.g. info, debug, warning. Default: INFO")

    parser.add_argument(
        "--artist",
        default=None,
        help="Optional: song artist for lyrics lookup and auto-correction",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional: song title for lyrics lookup and auto-correction",
    )

    parser.add_argument(
        "--genius_api_token",
        default=None,
        help="Optional: Genius API token for lyrics fetching. Can also be set with GENIUS_API_TOKEN env var.",
    )
    parser.add_argument(
        "--spotify_cookie",
        default=None,
        help="Optional: Spotify sp_dc cookie value for lyrics fetching. Can also be set with SPOTIFY_COOKIE_SP_DC env var.",
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

    log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)

    if not hasattr(args, "input_path"):
        parser.print_help()
        exit(1)

    logger.info(f"Karaoke generator beginning with input_path: {args.input_path}")

    generator = KaraokeGenerator(
        log_formatter=log_formatter,
        log_level=log_level,
        input_path=args.input_path,
        artist=args.artist,
        title=args.title,
        genius_api_token=args.genius_api_token,
        spotify_cookie=args.spotify_cookie,
        model_name=args.model_name,
        model_file_dir=args.model_file_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    outputs = generator.generate()

    logger.info(f"Karaoke generation complete!")

    logger.debug(f"Output folder: {outputs['output_dir']}")


if __name__ == "__main__":
    main()
