import os, re, subprocess
from difflib import SequenceMatcher

from utilities import conf


def process_transcripts():
    download_all_transcriptions()


def download_all_transcriptions():
    transcription_dir = conf.get("transcription_directory")
    if not os.path.exists(transcription_dir):
        os.makedirs(transcription_dir)

    all_reactions = list(conf.get("reactions").keys())
    all_transcripts = []
    for i, channel in enumerate(all_reactions):
        if channel == "Resound":
            continue

        if i < 18 or i > 40:
            continue

        reaction = conf.get("reactions").get(channel)

        channel = reaction.get("channel")
        vid = reaction.get("vid")

        transcript = create_transcription(video_id=vid, channel=channel)
        if transcript is not None:
            all_transcripts.append([channel, transcript])

        print(channel)

    lyrics_file = os.path.join(conf.get("song_directory"), "lyrics.txt")
    has_lyrics = os.path.exists(lyrics_file)

    # output to markdown
    output_path = os.path.join(transcription_dir, "_consolidated_transcripts.html")
    f = open(output_path, "w")
    f.write(
        f"""<html><body><div><div><h1>Reaction video transcripts for {conf.get('song_name')} by {conf.get('artist')}</h1>

        <p>This document contains {len(all_transcripts)} transcripts of 
        reactions to the song "{conf.get('song_name')}" by {conf.get('artist')}.</p> 

        <p>Each list item is a reaction.</p>

        <p>Each transcript also contains timing information for when in the video the words were expressed.</p>"""
    )

    if has_lyrics:
        f.write(
            "<p>This song has vocals, so some of the song lyrics might be repeated in the transcript.</p>"
        )
    f.write("</div><ul>")

    for i, (channel, transcript) in enumerate(all_transcripts):
        f.write("\n\n<li>")
        f.write(f"<h2>{i+1}. {channel}</h2>")
        f.write("\n\n")

        f.write(f"<div>{transcript}</div>")
        f.write("\n</li>")
    f.write("</ul></div></body></html>")

    f.close()
    print("output file")


def create_transcription(video_id, channel):
    song_key = conf.get("song_key")

    transcription_dir = conf.get("transcription_directory")

    raw_transcript = os.path.join(transcription_dir, f"{channel}.en.vtt")
    escped_path = os.path.join(transcription_dir, channel)
    if False and not os.path.exists(raw_transcript):
        cmd = f"yt-dlp --write-auto-subs --skip-download 'https://www.youtube.com/watch?v={video_id}' -o \"{escped_path}.%(ext)s\""
        subprocess.run(cmd, shell=True, check=True)
        # time.sleep(5)

    if os.path.exists(raw_transcript):
        cleaned_transcript_file = os.path.join(transcription_dir, f"{channel}.txt")
        if not os.path.exists(cleaned_transcript_file):
            clean_vtt(file_path=raw_transcript, output_path=cleaned_transcript_file)

        cleaned_transcript = open(cleaned_transcript_file, "r").read()

    else:
        print(f"ERROR! No transcript found for {channel}")
        return None
    return cleaned_transcript


def clean_vtt(file_path, output_path, interval=30):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    cleaned_lines = []
    last_timestamp = None
    buffer_lines = []
    current_start_time = None
    end_time = None
    last_line = ""

    for line in lines:
        # Extract timestamps
        timestamp_match = re.search(
            r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})", line
        )
        if timestamp_match:
            start_time = timestamp_match.group(1)
            end_time = timestamp_match.group(2)
            start_seconds = convert_to_seconds(start_time)

            if last_timestamp is None or start_seconds - last_timestamp >= interval:
                if buffer_lines:
                    cleaned_lines.append(
                        f"<br><br><span>{current_start_time} --> {end_time}<span><br>"
                    )
                    cleaned_lines.extend(buffer_lines)
                    cleaned_lines.append("")  # Add a blank line for separation
                    buffer_lines = []
                last_timestamp = start_seconds
                current_start_time = start_time
            continue  # Skip writing the timestamp line directly

        # Remove timestamp tags within the text
        line = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}><c>", "", line)
        line = re.sub(r"</c>", "", line)
        # Strip leading and trailing whitespace
        line = line.strip()
        if line and line != last_line:
            buffer_lines.append(line)
            last_line = line

    # Add any remaining buffered lines
    if buffer_lines and current_start_time:
        cleaned_lines.append(
            f"<br><br><span>{current_start_time} --> {end_time}<span><br>"
        )
        cleaned_lines.extend(buffer_lines)
        cleaned_lines.append("")  # Add a blank line for separation

    cleaned_lines = "\n".join(cleaned_lines[5:]).replace("&nbsp;", " ")

    # cleaned_transcript = re.sub(
    #     r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})",
    #     r"\n\n<span>\g<0></span>",
    #     cleaned_transcript,
    # )

    # cleaned_transcript = (
    #     cleaned_transcript.replace("WEBVTT", "")
    #     .replace("Kind: captions", "")
    #     .replace("Language: en", "")

    # )

    # cleaned_transcript = "\n".join(cleaned_transcript.split("\n")[2:])

    lyrics_file = os.path.join(conf.get("song_directory"), "lyrics.txt")
    has_lyrics = os.path.exists(lyrics_file)
    if has_lyrics:
        lyrics = load_lyrics(lyrics_file)
        cleaned_lines = scrub_lyrics_from_transcript(
            cleaned_lines, lyrics, similarity_threshold=0.5
        )

    # Write cleaned transcript to a new file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(cleaned_lines)

    return cleaned_lines


def convert_to_seconds(timestamp):
    # Split the timestamp into components
    parts = re.split("[:.]", timestamp)
    h, m, s = map(int, parts[:3])
    ms = int(parts[3])
    return h * 3600 + m * 60 + s + ms / 1000


def load_lyrics(lyrics_file):
    with open(lyrics_file, "r", encoding="utf-8") as file:
        lyrics = file.readlines()
    # Clean lyrics
    lyrics = [
        re.sub(r"[^\w\s]", "", line).strip().lower() for line in lyrics if line.strip()
    ]
    return lyrics


def scrub_lyrics_from_transcript(transcript, lyrics, similarity_threshold=0.5):
    def is_similar(a, b):
        return SequenceMatcher(None, a, b).ratio() > similarity_threshold

    scrubbed_transcript = []
    for line in transcript:
        clean_line = re.sub(r"[^\w\s]", "", line).strip().lower()
        if not any(is_similar(clean_line, lyric) for lyric in lyrics):
            scrubbed_transcript.append(line)

    return "".join(scrubbed_transcript)
