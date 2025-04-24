import os
from dotenv import load_dotenv
from openai import OpenAI
import whisper
import torch
import pyttsx3
import sounddevice as sd
from scipy.io.wavfile import write, read as read_wav
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk, ImageSequence

# === Load API Key ===
load_dotenv()
client = OpenAI()

# === Configuration ===
DURATION = 5
SAMPLE_RATE = 16000
FILENAME = os.path.join(os.getcwd(), "voice_input.wav")
GIF_PATH = "sakura7.gif"  

model = whisper.load_model("base")
engine = pyttsx3.init()

# === Voice Assistant Logic ===
def run_assistant():
    
    #Task 1: Records audio from your microphone
    status_label.config(text="🌸 Listening...", fg="#d6336c")
    root.update()
    
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    write(FILENAME, SAMPLE_RATE, recording)

    #Task 2: Converts audio to text using Whisper
    status_label.config(text="⛅ Transcribing...")
    root.update()

    _, audio_data = read_wav(FILENAME)
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32) / 32768.0
    audio_tensor = whisper.audio.pad_or_trim(audio_tensor)
    mel = whisper.log_mel_spectrogram(audio_tensor).to(model.device)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    transcribed_text = result.text.strip()

    #Task 3: Sends transcribed text to ChatGPT
    status_label.config(text="🪷 Asking ChatGPT...")
    root.update()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": transcribed_text}]
    )
    reply = response.choices[0].message.content.strip()

    #Task 4: Displays original query and GPT response
    transcript_box.delete(1.0, tk.END)
    transcript_box.insert(tk.END, f"🌸 You said:\n{transcribed_text}\n\n🤖 GPT replied:\n{reply}")
    status_label.config(text="🌼 Ready", fg="#8d448b")

    #Task 5: Speaks GPT-generated reply
    engine.say(reply)
    engine.runAndWait()

# === Animate GIF Background ===
def animate_gif(frame_idx=0):
    frame = frames[frame_idx]
    background.itemconfig(bg_img_id, image=frame)
    root.after(100, animate_gif, (frame_idx + 1) % len(frames))

# === GUI Setup ===
root = tk.Tk()
root.title("Sakura Voice Assistant")
root.geometry("700x500")
root.resizable(False, False)

# Load and prepare GIF frames
gif = Image.open(GIF_PATH)
frames = [ImageTk.PhotoImage(frame.copy().convert("RGBA")) for frame in ImageSequence.Iterator(gif)]

# Create background canvas
background = Canvas(root, width=480, height=460, highlightthickness=0)
background.place(x=0, y=0, relwidth=1, relheight=1)
bg_img_id = background.create_image(0, 0, anchor=tk.NW, image=frames[0])
animate_gif()

# GUI overlay widgets
frame = tk.Frame(root, bg="#ffffff", bd=0)
frame.place(relx=0.5, rely=0.1, anchor="n")

#Title
title = tk.Label(root, text="🌸 Sakura Voice Assistant", font=("Segoe UI", 18, "bold"), bg="#ffffff", fg="#8d448b")
title.place(relx=0.5, rely=0.05, anchor="n")

#Lable
status_label = tk.Label(root, text="Click the blossom to begin 🌸", font=("Segoe UI", 12), bg="#ffffff", fg="#d6336c")
status_label.place(relx=0.5, rely=0.15, anchor="n")

#Button
speak_button = tk.Button(
    root,
    text="🌸 Speak 🌸",
    font=("Segoe UI", 14, "bold"),
    bg="#ffb7c5",
    fg="#8d448b",
    activebackground="#f78da7",
    activeforeground="#ffffff",
    bd=0,
    padx=25,
    pady=10,
    command=run_assistant
)
speak_button.place(relx=0.5, rely=0.25, anchor="n")

#Text
transcript_box = tk.Text(
    root,
    height=10,
    width=50,
    font=("Segoe UI", 11),
    bg="#fffafa",
    fg="#4d2c4b",
    wrap="word",
    relief="flat",
    borderwidth=5
)
transcript_box.place(relx=0.5, rely=0.4, anchor="n")
transcript_box.insert(tk.END, "📝 Transcripts and replies will appear here...")

root.mainloop()