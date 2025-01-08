import whisper

# Load the model
model = whisper.load_model("base")

# Transcribe the audio
names = ["imagine_bad_beaut_75"]
for name in names:
    result = model.transcribe(f"{name}.wav")
    predicted_text = result["text"]
    with open(f"{name}.txt", 'w') as f:
        f.write(predicted_text)

# result = model.transcribe("imagine_bad_beaut_25.wav")
# predicted_text = result["text"]
# print(f"Transcribed Text: {predicted_text}")

# result = model.transcribe("imagine_bad_beaut_50.wav")
# predicted_text = result["text"]
# print(f"Transcribed Text: {predicted_text}")

# result = model.transcribe("imagine_bad_beaut_75.wav")
# predicted_text = result["text"]
# print(f"Transcribed Text: {predicted_text}")

# result = model.transcribe("imagine_bad_beaut_100.wav")
# predicted_text = result["text"]
# print(f"Transcribed Text: {predicted_text}")
