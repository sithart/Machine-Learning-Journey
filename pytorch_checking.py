import whisper
audio_path = r'Tamil Cut Songs - Onakkaga poranthenae.mp3'
model = whisper.load_model("base")
result = model.transcribe(audio_path)
print(result["text"])