# KaraokeHunt: Karaoke video generator
Fully automated creation of _acceptable_ karaoke music videos from any music on YouTube, using open source tools and AI (e.g. Whisper and MDX-Net)

## Context
This is one experimental tool as part of the journey towards implementing the [full vision](https://docs.google.com/document/d/19LS1aJI8YwSmkWmDdpCHpmTGiHL9l0VDJ1SxSl4l6Z8/edit#) for KaraokeHunt (https://karaokehunt.com).

Some of the other components include:
- [Lyrics from Genius](https://github.com/karaokenerds/lyrics-from-genius)
- [Karaoke Song Finder (Spreadsheet generator)](https://github.com/karaokenerds/music-data-karaoke-song-sheets)
- [KaraokeHunt Mobile App](https://github.com/karaokenerds/karaokehunt-app)

## Idea steps
- Fetch the requested YouTube video using [yt-dlp](https://github.com/yt-dlp/yt-dlp) and extract the audio to wav using ffmpeg
- Run that audio through an ML-based vocal isolation model tuned for karaoke (e.g. [UVR-MDX-NET Karaoke 2](https://github.com/Anjok07/ultimatevocalremovergui/blob/master/models/MDX_Net_Models/model_data/model_name_mapper.json#L12) to get high quality instrumental audio without lead vocals but retaining backing vocals
- Run the lead vocal track through [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) to generate a time-synced lyrics file
- Correct the detected lyrics by fetching lyrics from a human-input source (e.g. musicxmatch/spotify using [syrics](https://github.com/akashrchandran/syrics), genius using [lyrics-from-genius](https://github.com/karaokenerds/lyrics-from-genius) and attempting to match up segments with the whisper-heard lyrics whilst maintaining timestamps
  - Potentially also consider splitting words by syllable (e.g. using [python-syllables](https://github.com/prosegrinder/python-syllables) and attempting to guess the sub-word timestamps 
- Generate a new video file using the instrumental audio and a background image, with the synced lyrics “burned” into the video at the correct timestamps
  - Lots of scope to make this really nice, e.g. adjusting kerning dynamically to fit longer lines on one screen, but also lots of gotchas e.g. super long lines needing to be split at a reasonable place
- Publish this video to YouTube
