"""Chat bot UI for a local Ollama model (gemma3:4b).

Features:
  * streaming replies
  * Markdown-style formatting (bold, italic, `code`, code blocks, headings, bullets)
  * skill.md loading (the skill text is used as the system prompt)
  * Stop button / Esc key to cancel a reply while the model is thinking or writing

Setup:
    ollama pull gemma3:4b
    pip install -r requirements.txt
    python chatbot.py
"""

import queue
import re
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

import ollama

MODEL = "gemma3:4b"
SKILL_DIR = Path(__file__).resolve().parent
DEFAULT_SKILL = SKILL_DIR / "skill.md"  # loaded automatically if it exists

# **bold**, `code`, *italic*
INLINE = re.compile(r"\*\*.+?\*\*|`[^`]+`|\*[^*\s][^*]*?\*")


def load_skill(path):
    """Return (name, body) from a skill.md file.

    Optional YAML-style front matter is supported:
        ---
        name: My Skill
        description: What it does
        ---
    The body (everything after the front matter) becomes the system prompt.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    name, body = path.stem, text
    if text.startswith("---"):
        parts = text.split("---", 2)  # ['', front matter, body]
        if len(parts) == 3:
            for line in parts[1].splitlines():
                key, _, value = line.partition(":")
                if key.strip().lower() == "name" and value.strip():
                    name = value.strip()
            body = parts[2].strip()
    return name, body


class ChatApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"Chat with {MODEL}")
        self.root.geometry("1000x800")

        self.messages = []             # user/assistant history (no system prompt)
        self.skill_name = None
        self.skill_body = None

        self.chunks = queue.Queue()    # (run_id, kind, data) from the worker thread
        self.run_id = 0
        self.stop_event = threading.Event()
        self.busy = False
        self.reply = ""                # full text of the streaming reply
        self.in_code = False           # inside a ``` block

        sans = tkfont.nametofont("TkDefaultFont").actual("family")
        mono = tkfont.nametofont("TkFixedFont").actual("family")

        # --- skill bar -----------------------------------------------------
        bar = tk.Frame(root)
        bar.pack(fill=tk.X, padx=8, pady=(8, 0))
        self.skill_label = tk.Label(bar, text="Skill: none", anchor="w")
        self.skill_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(bar, text="Load skill…", command=self.pick_skill).pack(side=tk.LEFT)
        tk.Button(bar, text="Unload", command=self.unload_skill).pack(side=tk.LEFT, padx=(6, 0))

        # --- chat display --------------------------------------------------
        c = self.chat = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state=tk.DISABLED, font=(sans, 11), padx=8, pady=8
        )
        c.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
        # Tag order matters: later tags win when they set the same option
        # (so a heading stays a heading even if it contains **bold**).
        c.tag_config("you", foreground="#1a5fb4", font=(sans, 11, "bold"))
        c.tag_config("bot", foreground="#26a269", font=(sans, 11, "bold"))
        c.tag_config("error", foreground="#c01c28")
        c.tag_config("note", foreground="#6b6b6b", font=(sans, 10, "italic"))
        c.tag_config("bold", font=(sans, 11, "bold"))
        c.tag_config("italic", font=(sans, 11, "italic"))
        c.tag_config("code", font=(mono, 10), background="#eceff1")
        c.tag_config("codeblock", font=(mono, 10), background="#eceff1", lmargin1=16, lmargin2=16)
        c.tag_config("h1", font=(sans, 16, "bold"), spacing1=6)
        c.tag_config("h2", font=(sans, 14, "bold"), spacing1=4)
        c.tag_config("h3", font=(sans, 12, "bold"))
        c.tag_config("bullet", lmargin1=12, lmargin2=26)
        c.mark_set("reply_start", "end-1c")
        c.mark_gravity("reply_start", tk.LEFT)

        # --- status line ---------------------------------------------------
        self.status = tk.Label(root, text="", anchor="w", fg="#6b6b6b")
        self.status.pack(fill=tk.X, padx=10)

        # --- input row -----------------------------------------------------
        row = tk.Frame(root)
        row.pack(fill=tk.X, padx=8, pady=(0, 8))
        # Multi-line input: Enter sends, Shift+Enter inserts a line break.
        self.entry = tk.Text(row, height=5, wrap=tk.WORD, font=(sans, 11), padx=4, pady=4)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", self.on_enter)
        self.entry.bind("<Shift-Return>", self.on_shift_enter)
        self.entry.focus()
        self.send_btn = tk.Button(row, text="Send", width=8, command=self.send)
        self.send_btn.pack(side=tk.LEFT, padx=(6, 0), anchor=tk.S)
        self.stop_btn = tk.Button(row, text="Stop", width=8, command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(6, 0), anchor=tk.S)
        tk.Button(row, text="Clear", width=8, command=self.clear).pack(
            side=tk.LEFT, padx=(6, 0), anchor=tk.S
        )
        self.root.bind("<Escape>", self.stop)

        if DEFAULT_SKILL.exists():
            self.set_skill(DEFAULT_SKILL)

    # ---- skills --------------------------------------------------------
    def pick_skill(self):
        path = filedialog.askopenfilename(
            title="Load skill",
            initialdir=SKILL_DIR,
            filetypes=[("Markdown", "*.md"), ("All files", "*.*")],
        )
        if path:
            self.set_skill(path)

    def set_skill(self, path):
        try:
            name, body = load_skill(path)
        except (OSError, UnicodeDecodeError) as exc:
            messagebox.showerror("Skill", f"Could not read the skill file:\n{exc}")
            return
        if not body:
            messagebox.showwarning("Skill", "That skill file is empty.")
            return
        self.skill_name, self.skill_body = name, body
        self.skill_label.config(text=f"Skill: {name}")
        self.append(f"Skill loaded: {name} (applies from your next message)\n\n", "note")

    def unload_skill(self):
        if self.skill_body is None:
            return
        self.skill_name = self.skill_body = None
        self.skill_label.config(text="Skill: none")
        self.append("Skill unloaded (applies from your next message)\n\n", "note")

    # ---- display helpers -----------------------------------------------
    def append(self, text, *tags):
        c = self.chat
        c.config(state=tk.NORMAL)
        c.insert(tk.END, text, tags)
        c.see(tk.END)
        c.config(state=tk.DISABLED)

    def clear(self):
        if self.busy:
            return
        self.messages.clear()
        self.chat.config(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.config(state=tk.DISABLED)

    # ---- Markdown rendering (widget must be NORMAL when these run) -----
    def inline(self, text, base=()):
        c = self.chat
        pos = 0
        for m in INLINE.finditer(text):
            if m.start() > pos:
                c.insert(tk.END, text[pos:m.start()], base)
            tok = m.group(0)
            if tok.startswith("**"):
                c.insert(tk.END, tok[2:-2], base + ("bold",))
            elif tok.startswith("`"):
                c.insert(tk.END, tok[1:-1], base + ("code",))
            else:
                c.insert(tk.END, tok[1:-1], base + ("italic",))
            pos = m.end()
        if pos < len(text):
            c.insert(tk.END, text[pos:], base)

    def render_line(self, line):
        c = self.chat
        if line.strip().startswith("```"):
            self.in_code = not self.in_code
            return
        if self.in_code:
            c.insert(tk.END, line + "\n", "codeblock")
            return
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            tag = f"h{len(m.group(1))}"
            self.inline(m.group(2), (tag,))
            c.insert(tk.END, "\n", tag)
            return
        m = re.match(r"^\s*[-*+]\s+(.*)", line)
        if m:
            c.insert(tk.END, "• ", "bullet")
            self.inline(m.group(1), ("bullet",))
            c.insert(tk.END, "\n", "bullet")
            return
        m = re.match(r"^\s*(\d+\.)\s+(.*)", line)
        if m:
            c.insert(tk.END, m.group(1) + " ", "bullet")
            self.inline(m.group(2), ("bullet",))
            c.insert(tk.END, "\n", "bullet")
            return
        self.inline(line)
        c.insert(tk.END, "\n")

    def redraw(self, final=False):
        """Render the reply while preserving real and escaped line breaks."""
        c = self.chat
        c.config(state=tk.NORMAL)
        c.delete("reply_start", tk.END)
        self.in_code = False

        text = self.reply.replace("\\r\\n", "\\n").replace("\\r", "\\n")

        # Some local models return the characters backslash+n instead of a
        # real newline. Convert those when the response contains no real
        # line breaks, so Markdown paragraphs and lists don't run together.
        if "\\n" not in text and "\\\\n" in text:
            text = text.replace("\\\\n", "\\n").replace("\\\\r", "\\n")

        lines = text.split("\\n")
        if final and lines and lines[-1] == "":
            lines.pop()

        for i, line in enumerate(lines):
            # Render all complete lines as Markdown. During streaming, leave
            # only the final incomplete line plain until more tokens arrive.
            if i == len(lines) - 1 and not final:
                c.insert(tk.END, line)
            else:
                self.render_line(line)

        self.in_code = False
        c.see(tk.END)
        c.config(state=tk.DISABLED)

    # ---- sending / streaming -------------------------------------------
    def on_enter(self, event=None):
        self.send()
        return "break"  # don't insert a newline

    def on_shift_enter(self, event=None):
        self.entry.insert(tk.INSERT, "\n")
        return "break"

    def send(self):
        text = self.entry.get("1.0", "end-1c").strip()
        if not text or self.busy:
            return
        self.entry.delete("1.0", tk.END)
        self.messages.append({"role": "user", "content": text})
        history = list(self.messages)
        if self.skill_body:
            history.insert(0, {"role": "system", "content": self.skill_body})

        self.append("You: ", "you")
        self.append(text + "\n\n")
        self.append(f"{MODEL}:\n", "bot")
        self.chat.mark_set("reply_start", "end-1c")

        self.reply, self.in_code = "", False
        self.run_id += 1
        self.stop_event = threading.Event()
        self.busy = True
        self.send_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status.config(text="Thinking…  (Esc or Stop to cancel)")

        threading.Thread(
            target=self.stream_reply,
            args=(self.run_id, history, self.stop_event),
            daemon=True,
        ).start()
        self.root.after(30, self.poll)

    def stream_reply(self, run_id, history, stop_event):
        """Worker thread: talks to Ollama and passes chunks to the UI via a queue."""
        stream = None
        try:
            stream = ollama.chat(model=MODEL, messages=history, stream=True)
            for part in stream:
                if stop_event.is_set():
                    break
                self.chunks.put((run_id, "text", part["message"]["content"]))
        except Exception as exc:
            self.chunks.put((run_id, "error", str(exc)))
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        self.chunks.put((run_id, "done", ""))

    def poll(self):
        """Runs on the Tk thread: moves queued chunks into the widget."""
        changed, outcome, error = False, None, None
        while outcome is None:
            try:
                run_id, kind, data = self.chunks.get_nowait()
            except queue.Empty:
                break
            if run_id != self.run_id:  # leftovers from a stopped reply
                continue
            if kind == "text":
                if not self.reply:
                    self.status.config(text="Writing…  (Esc or Stop to cancel)")
                self.reply += data
                changed = True
            elif kind == "error":
                outcome, error = "error", data
            else:
                outcome = "done"
        if changed and outcome is None:
            self.redraw()  # one redraw per batch of chunks
        if outcome == "error":
            self.finish(error=error)
        elif outcome == "done":
            self.finish()
        elif self.busy:
            self.root.after(30, self.poll)

    def stop(self, event=None):
        """Cancel the current reply immediately (also works while it is still thinking)."""
        if self.busy:
            self.stop_event.set()
            self.finish(stopped=True)

    def finish(self, stopped=False, error=None):
        if not self.busy:
            return
        self.busy = False
        self.redraw(final=True)
        if self.reply.strip():  # keep whatever was written, even if cut short
            self.messages.append({"role": "assistant", "content": self.reply})
        else:                   # no answer: forget the unanswered question
            self.messages.pop()
        if error:
            self.append(
                f"\n[Error: {error}]\nIs Ollama running and is '{MODEL}' pulled?\n\n", "error"
            )
        elif stopped:
            self.append("\n[stopped]\n\n", "note")
        else:
            self.append("\n")
        self.send_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status.config(text="")
        self.entry.focus()


if __name__ == "__main__":
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()