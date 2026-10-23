"""Simple chat bot UI for a local Ollama model (gemma3:4b).

Setup:
    ollama pull gemma3:4b
    pip install ollama
    python chatbot.py
"""

import queue
import threading
import tkinter as tk
from tkinter import scrolledtext

import ollama

MODEL = "gemma3:4b"


class ChatApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"Chat with {MODEL}")
        self.root.geometry("640x520")

        self.messages = []          # conversation history sent to the model
        self.chunks = queue.Queue() # streamed text from the worker thread
        self.busy = False

        # Chat display
        self.chat = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state=tk.DISABLED, font=("Helvetica", 11), padx=8, pady=8
        )
        self.chat.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
        self.chat.tag_config("you", foreground="#1a5fb4", font=("Helvetica", 11, "bold"))
        self.chat.tag_config("bot", foreground="#26a269", font=("Helvetica", 11, "bold"))
        self.chat.tag_config("error", foreground="#c01c28")

        # Input row
        row = tk.Frame(root)
        row.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.entry = tk.Entry(row, font=("Helvetica", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.entry.bind("<Return>", lambda e: self.send())
        self.entry.focus()
        self.send_btn = tk.Button(row, text="Send", width=8, command=self.send)
        self.send_btn.pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(row, text="Clear", width=8, command=self.clear).pack(side=tk.LEFT, padx=(6, 0))

    # ---- helpers -------------------------------------------------------
    def append(self, text, tag=None):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text, tag)
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def set_busy(self, busy):
        self.busy = busy
        self.send_btn.config(state=tk.DISABLED if busy else tk.NORMAL)

    def clear(self):
        if self.busy:
            return
        self.messages.clear()
        self.chat.config(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.config(state=tk.DISABLED)

    # ---- sending / streaming ------------------------------------------
    def send(self):
        text = self.entry.get().strip()
        if not text or self.busy:
            return
        self.entry.delete(0, tk.END)
        self.messages.append({"role": "user", "content": text})
        self.append("You: ", "you")
        self.append(text + "\n\n")
        self.append(f"{MODEL}: ", "bot")
        self.set_busy(True)
        threading.Thread(target=self.stream_reply, daemon=True).start()
        self.root.after(30, self.poll)

    def stream_reply(self):
        reply = ""
        try:
            for part in ollama.chat(model=MODEL, messages=self.messages, stream=True):
                piece = part["message"]["content"]
                reply += piece
                self.chunks.put(("text", piece))
            self.messages.append({"role": "assistant", "content": reply})
        except Exception as exc:
            self.messages.pop()  # drop the user message that got no answer
            self.chunks.put(("error", f"\n[Error: {exc}]\nIs Ollama running and is "
                                      f"'{MODEL}' pulled?"))
        self.chunks.put(("done", ""))

    def poll(self):
        """Runs on the Tk thread and moves streamed text into the widget."""
        finished = False
        while True:
            try:
                kind, data = self.chunks.get_nowait()
            except queue.Empty:
                break
            if kind == "text":
                self.append(data)
            elif kind == "error":
                self.append(data, "error")
            else:
                finished = True
        if finished:
            self.append("\n\n")
            self.set_busy(False)
        else:
            self.root.after(30, self.poll)


if __name__ == "__main__":
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()
