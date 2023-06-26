#!/usr/bin/env python
import sys
import subprocess
import whisper_timestamped as whisper

def calculate_singing_percentage(audio_filename):
    audio = whisper.load_audio(audio_filename)
    model = whisper.load_model("medium.en", device="cpu")

    result = whisper.transcribe(model, audio, language="en")

    # Calculate total seconds of singing
    total_singing_duration = sum(segment["end"] - segment["start"] for segment in result["segments"])

    # Calculate song duration using ffprobe
    duration_command = ['ffprobe', '-i', audio_filename, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")]
    duration_output = subprocess.check_output(duration_command, universal_newlines=True)
    song_duration = float(duration_output)

    # Calculate singing percentage
    singing_percentage = (total_singing_duration / song_duration) * 100

    return singing_percentage


if len(sys.argv) < 2:
    print("Please provide the audio file name as the first command-line parameter.")
    sys.exit(1)

audio_filename = sys.argv[1]
singing_percentage = calculate_singing_percentage(audio_filename)
print("Singing percentage: {:.2f}%".format(singing_percentage))
