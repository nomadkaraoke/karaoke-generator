#!/usr/bin/env python
import sys
import subprocess
import json
import os
import whisper_timestamped as whisper

CACHE_FOLDER = ".cache"

def load_transcription_result(audio_filename):
    cache_filename = get_cache_filename(audio_filename)

    if os.path.isfile(cache_filename):
        with open(cache_filename, "r") as cache_file:
            return json.load(cache_file)

    audio = whisper.load_audio(audio_filename)
    model = whisper.load_model("medium.en", device="cpu")
    result = whisper.transcribe(model, audio, language="en")

    with open(cache_filename, "w") as cache_file:
        json.dump(result, cache_file)

    return result


def calculate_singing_percentage(audio_filename):
    result = load_transcription_result(audio_filename)

    # Calculate total seconds of singing
    total_singing_duration = sum(segment["end"] - segment["start"] for segment in result["segments"])

    # Calculate song duration using ffprobe
    duration_command = ['ffprobe', '-i', audio_filename, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")]
    duration_output = subprocess.check_output(duration_command, universal_newlines=True)
    song_duration = float(duration_output)

    # Calculate singing percentage
    singing_percentage = (total_singing_duration / song_duration) * 100

    return singing_percentage


def get_cache_filename(audio_filename):
    filename = os.path.splitext(audio_filename)[0]
    cache_filename = os.path.join(CACHE_FOLDER, filename + ".json")
    return cache_filename


def create_cache_folder():
    if not os.path.exists(CACHE_FOLDER):
        os.makedirs(CACHE_FOLDER)


if len(sys.argv) < 2:
    print("Please provide the audio file name as the first command-line parameter.")
    sys.exit(1)

audio_filename = sys.argv[1]
create_cache_folder()
singing_percentage = calculate_singing_percentage(audio_filename)
print("Singing percentage: {:.2f}%".format(singing_percentage))
