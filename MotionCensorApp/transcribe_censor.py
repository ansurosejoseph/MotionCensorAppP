import whisper
import sys
import ffmpeg
import os
import json

# List of inappropriate words to censor (add more as needed)
inappropriate_words = {"fuck",  "badword3","pervert","bitch"}

def estimate_word_timestamps(segment):
    """Estimate timestamps for words within a segment."""
    start_time = segment['start']
    end_time = segment['end']
    text = segment['text'].strip()
    words = text.split()

    # Calculate duration per word
    segment_duration = end_time - start_time
    num_words = len(words)
    if num_words == 0:
        return []

    duration_per_word = segment_duration / num_words

    # Create estimated timestamps for each word
    word_timestamps = []
    current_time = start_time
    for word in words:
        word_start = current_time
        word_end = current_time + duration_per_word
        word_timestamps.append((word, word_start, word_end))
        current_time = word_end

    return word_timestamps

def transcribe_and_censor(video_path, output_folder):
    print(f"Video Path: {video_path}")
    print(f"Output Folder: {output_folder}")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define paths for output files
    output_path = os.path.join(output_folder, "censored_output.mp4")
    transcription_path = os.path.join(output_folder, "transcription.json")

    # Load the Whisper model
    model = whisper.load_model("small")
    print("Whisper model loaded successfully.")

    # Transcribe the video
    print("Starting transcription...")
    result = model.transcribe(video_path, language='en')
    print("Transcription completed.")

    # Save the transcription result to a JSON file
    with open(transcription_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    print(f"Transcription saved to: {transcription_path}")

    segments = result.get('segments', [])
    silence_intervals = []

    # Identify timestamps for inappropriate words
    for segment in segments:
        word_timestamps = estimate_word_timestamps(segment)
        for word, start, end in word_timestamps:
            if word.lower().strip(".,!?") in inappropriate_words:
                # Add a small buffer of 0.1 seconds before and after the word
                silence_intervals.append((max(0, start - 0.05), end + 0.05))
                print(f"Found inappropriate word '{word}' at {start:.2f} - {end:.2f}")

    # Check if any inappropriate words were found
    if not silence_intervals:
        print("No inappropriate words found.")
        return

    # Use FFmpeg to mute the identified sections
    try:
        print("Starting FFmpeg processing...")
        input_stream = ffmpeg.input(video_path)
        audio = input_stream.audio

        for start, end in silence_intervals:
            print(f"Muting section from {start:.2f} to {end:.2f}")
            audio = ffmpeg.filter(audio, 'volume', '0', enable=f'between(t,{start},{end})')

        # Output the final censored video
        ffmpeg.output(input_stream.video, audio, output_path).run(overwrite_output=True)
        print(f"Censored video saved successfully at: {output_path}")

    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python transcribe_censor.py <input_video_path> <output_folder>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_folder = sys.argv[2]

    transcribe_and_censor(video_path, output_folder)
