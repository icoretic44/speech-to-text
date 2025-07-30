from faster_whisper import WhisperModel
import config

class Transcriber:
    def __init__(self):
        self.model = WhisperModel(
            config.MODEL_SIZE,  
            device=config.DEVICE,     
            compute_type="float16" 
        )

    def transcribe_audio(self, audio_data):
        """
        Transcribes the given audio data using the faster-whisper model.
        """
        segments, info = self.model.transcribe(
            audio_data,           # The NumPy array with the audio data.
            beam_size=5,          # A parameter for the transcription algorithm.
            language=config.LANGUAGE, # The language we expect.
            vad_filter=True, 
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        print(f"Detected language '{info.language}' with probability {info.language_probability}")

        transcribed_text = ""
        for segment in segments:
            transcribed_text += segment.text + " "
        
        # We use .strip() to remove any leading or trailing whitespace from the
        # final combined string before returning it.
        return transcribed_text.strip()