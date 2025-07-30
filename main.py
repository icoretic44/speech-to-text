import queue              
import threading          
import numpy as np        
from AudioRecorder import AudioRecorder # Our custom "Ear" class
from Transcriber import Transcriber      # Our custom "Brain" class
import config             # Our application settings
import time               # To handle timing for silence detection
from scipy.io.wavfile import write  # To save audio files if needed
def main():
    # Create the shared "conveyor belt" queue.
    audio_queue = queue.Queue()

    # Create an instance of our recorder, passing it the shared queue.
    recorder = AudioRecorder(audio_queue)
    # Create an instance of our transcriber.
    transcriber = Transcriber()

    recording_thread = threading.Thread(target=recorder.start_recording)
    # Setting 'daemon' to True means this thread will automatically shut down
    # when the main program exits. 
    recording_thread.daemon = True
    recording_thread.start()

    print("Speak now. The application will transcribe your speech in real-time.")
    print("Press Ctrl+C to stop the application.")
    file_counter = 1
    # This 'try...finally' block ensures that our cleanup code runs
    # even if the user stops the program with Ctrl+C (KeyboardInterrupt).
    try:
        # Initialize variables for our silence detection logic.
        silence_start_time = None # When the current period of silence began.
        audio_buffer = []         # A list to hold audio chunks for one utterance.

        # This is the main loop of our application. It runs forever until stopped.
        while True:
            # This 'try...except' block handles getting data from the queue.
            try:
                # Get an audio chunk from the queue. 'block=False' means the program
                # won't wait if the queue is empty; it will raise a queue.Empty
                # exception instead, which we catch below.
                audio_chunk = audio_queue.get(block=False)
                # Add the new chunk to our temporary buffer.
                audio_buffer.append(audio_chunk)

                # --- Simple Silence Detection Logic ---
                # Check the loudest sound in the *current* chunk.
                if np.max(audio_chunk) < config.SILENCE_THRESHOLD:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    # If the timer *is* running, check if enough time has passed.
                    elif time.time() - silence_start_time > config.SILENCE_DURATION:
                        if audio_buffer:
                            full_audio = np.concatenate(audio_buffer)

                            # 1. FLATTEN THE ARRAY: Convert from (n, 1) to (n,).
                            processed_audio = full_audio.flatten()

                            # 2. NORMALIZE AND CONVERT TO FLOAT32:
                            #    Convert int16 range (-32768 to 32767) to float32 range (-1.0 to 1.0)
                            processed_audio = processed_audio.astype(np.float32) / 32768.0
                            text = transcriber.transcribe_audio(processed_audio)
                            
                            print(f"DEBUG: Raw text from transcriber: '{text}'")
                            print(f"Transcription: {text}")

                            # Clear buffer and reset silence timer
                            audio_buffer = []
                            silence_start_time = None
                else:
                    # If the chunk was loud (not silent), reset the silence timer.
                    # This means the user is still speaking.
                    silence_start_time = None

            except queue.Empty:
                continue

    except KeyboardInterrupt:
        # This block is executed if the user presses Ctrl+C.
        print("\nStopping application...")
    finally:
        recorder.stop_recording()
        print("Application stopped.")

if __name__ == "__main__":
    main()