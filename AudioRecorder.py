import sounddevice as sd  
import numpy as np         
import queue               
import config              

class AudioRecorder:
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        self.is_recording = False

    def _audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status)
        # 'indata' is a NumPy array containing the new audio data.
        # We put a copy of it into our shared queue for the main thread to process.
        # .copy() is important to prevent potential race conditions where the
        # data buffer is modified before the main thread can process it.
        self.audio_queue.put(indata.copy())

    def start_recording(self):
        """Starts the audio recording."""
        self.is_recording = True
        print("Recording started...")
        # sd.InputStream is a context manager that opens the microphone stream.
        with sd.InputStream(samplerate=config.SAMPLE_RATE,  
                              channels=config.CHANNELS,      
                              dtype=config.FORMAT,           
                              blocksize=config.CHUNK_SIZE,   
                              callback=self._audio_callback): 
            while self.is_recording:
                # sd.sleep() pauses this loop, preventing it from consuming CPU
                # while it waits. 100 milliseconds is a reasonable pause.
                sd.sleep(100)

    def stop_recording(self):
        """Stops the audio recording."""
        self.is_recording = False
        print("Recording stopped.")