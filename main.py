# config.json
{
    "audio_settings": {
        "duration": 0,  # 0 for unlimited recording
        "sample_rate": 16000
    },
    "model_configurations": {
        "whisper_model_size": "base",
        "device": "cpu",
        "pyannote_pipeline": "pyannote/speaker-diarization@2.1",
        "pyannote_auth_token": "YOUR_HF_TOKEN"
    },
    "file_paths": {
        "audio_output_dir": "audio/",
        "transcript_output_dir": "transcripts/"
    }
}

# main.py

import json
import threading
import queue
from record_audio import record_audio, listen_for_stop
from transcription import live_transcription
from diarization import process_audio
from utils import get_default_filename, prompt_filename

def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    config = load_config()

    stop_event = threading.Event()
    audio_queue = queue.Queue()
    transcription_queue = queue.Queue()

    audio_settings = config['audio_settings']
    model_config = config['model_configurations']
    file_paths = config['file_paths']

    # Prompt user to enter a name for the transcript file
    default_transcript = get_default_filename("meeting_minutes", "txt")
    transcript_filename = prompt_filename("Enter the name for the transcript file", default_transcript, ".txt")

    # Prompt user to enter a name for the audio file
    default_audio = get_default_filename("output", "wav")
    audio_filename = prompt_filename("Enter the name for the audio file", default_audio, ".wav")

    # Load Whisper model for live transcription
    print("Loading Whisper model for live transcription...")
    transcription_thread = threading.Thread(target=live_transcription, args=(audio_queue, transcription_queue, model_config))
    transcription_thread.start()

    recorder_thread = threading.Thread(target=record_audio, args=(stop_event, audio_queue, audio_filename, audio_settings))
    stopper_thread = threading.Thread(target=listen_for_stop, args=(stop_event,))

    recorder_thread.start()
    stopper_thread.start()

    recorder_thread.join()
    stopper_thread.join()
    transcription_thread.join()

    # Process the recorded audio and generate transcript
    process_audio(filename=audio_filename, transcript_filename=transcript_filename, config=config)

if __name__ == "__main__":
    main()

# record_audio.py

import pyaudio
import wave
import threading
import queue
import time
from utils import is_windows, get_char

def record_audio(stop_event, audio_queue, filename="output.wav", audio_settings=None):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # Mono recording for better processing
    RATE = audio_settings.get('sample_rate', 16000)  # 16kHz is suitable for Whisper

    p = pyaudio.PyAudio()

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []

        print("Recording... Press 's' to stop.")

        while not stop_event.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            audio_queue.put(data)

    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"Audio recorded and saved to {filename}")

def listen_for_stop(stop_event):
    try:
        if is_windows():
            import msvcrt  # For Windows
            print("Press 's' to stop recording.")
            while not stop_event.is_set():
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    if key == 's':
                        stop_event.set()
        else:
            import sys
            import select
            print("Press 's' then Enter to stop recording.")
            while not stop_event.is_set():
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if 's' in line.lower():
                        stop_event.set()
                time.sleep(0.1)
    except Exception as e:
        print(f"Error in listen_for_stop: {e}")
        stop_event.set()

# transcription.py

import whisper
import queue
import datetime
from utils import decode_audio_buffer

def live_transcription(audio_queue, transcription_queue, model_config):
    buffer = b''
    try:
        model = whisper.load_model(model_config.get('whisper_model_size', 'base')).to(model_config.get('device', 'cpu'))

        while True:
            try:
                data = audio_queue.get(timeout=1)
                buffer += data
                if len(buffer) >= model_config['audio_settings']['sample_rate'] * 2:  # Approximately 1 second of audio
                    audio_segment = decode_audio_buffer(buffer, model_config['audio_settings']['sample_rate'])
                    buffer = b''
                    result = model.transcribe(audio_segment, language="en", verbose=False)
                    transcription = result.get('text', '').strip()
                    if transcription:
                        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                        print(f"[{timestamp}] {transcription}")
                        transcription_queue.put(transcription)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue
    except Exception as e:
        print(f"Error in live_transcription: {e}")

# diarization.py

from pyannote.audio import Pipeline
import whisper
import datetime
from utils import time_format

def process_audio(filename="output.wav", transcript_filename="meeting_minutes.txt", config=None):
    try:
        # Initialize pyannote speaker diarization pipeline
        pipeline = Pipeline.from_pretrained(config['model_configurations']['pyannote_pipeline'],
                                            use_auth_token=config['model_configurations']['pyannote_auth_token'])
        diarization = pipeline(filename)

        # Load Whisper model
        model = whisper.load_model(config['model_configurations']['whisper_model_size']).to(config['model_configurations']['device'])

        # Transcribe with Whisper, enabling word timestamps
        transcription = model.transcribe(filename, word_timestamps=True)

        # Build a list of transcription segments
        transcript_segments = []
        for segment in transcription['segments']:
            transcript_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            })

        # Assign speakers to transcription segments
        script = []
        for transcript in transcript_segments:
            segment_start = transcript['start']
            segment_end = transcript['end']
            text = transcript['text']

            # Find the speaker for this segment
            speaker = "Unknown"
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                if turn.start <= segment_start < turn.end:
                    speaker = speaker_label
                    break

            script.append(f"{speaker} [{time_format(segment_start)} - {time_format(segment_end)}]: {text.strip()}")

        # Write the script to the transcript file
        with open(transcript_filename, "w") as f:
            for line in script:
                f.write(line + "\n")

        print(f"Meeting minutes have been saved to {transcript_filename}")

    except Exception as e:
        print(f"Error processing audio: {e}")

# format_script.py

# This file can be used for additional formatting if needed.
# Currently, formatting is handled within diarization.py.
# You can add functions here to further format the script as required.

def format_line(speaker, start_time, end_time, text):
    return f"{speaker} [{start_time} - {end_time}]: {text.strip()}"

# utils.py

import datetime
import sys
import select
import time
import numpy as np
from whisper.audio import decode_audio

def is_windows():
    return sys.platform.startswith('win')

def get_char():
    if is_windows():
        import msvcrt
        return msvcrt.getch().decode('utf-8').lower()
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            char = sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return char

def get_default_filename(base, extension):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base}_{timestamp}.{extension}"

def prompt_filename(prompt, default, extension):
    user_input = input(f"{prompt} (default: {default}): ").strip()
    if not user_input:
        return default
    elif not user_input.endswith(f".{extension}"):
        return f"{user_input}.{extension}"
    return user_input

def time_format(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def decode_audio_buffer(buffer, sample_rate):
    audio = np.frombuffer(buffer, np.int16).astype(np.float32) / 32768.0
    return decode_audio(audio, sample_rate)















