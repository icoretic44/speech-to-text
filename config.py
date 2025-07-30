#Whisper model configuration
MODEL_SIZE = "turbo"
LANGUAGE = "vi"
DEVICE = "cuda"

#Audio recording configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = "int16"

#Real-time processing configuration
SILENCE_THRESHOLD = 300
SILENCE_DURATION = 1