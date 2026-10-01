import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pather_g_optimized as pg
import threading


def xmla(folder_path):
    print(f"xmla called with: {folder_path}")
    pg.main(False, folder_path)



class FolderPicker(tk.Tk):
    def __init__(self, root_dir="."):
        super().__init__()
        self.title("Folder Picker")
        self.geometry("420x360")
        self.root_dir = os.path.abspath(root_dir)

        # Top bar: current directory + change button
        bar = tk.Frame(self)
        bar.pack(fill="x", padx=8, pady=6)
        self.path_var = tk.StringVar(value=self.root_dir)
        tk.Entry(bar, textvariable=self.path_var, state="readonly").pack(
            side="left", fill="x", expand=True
        )
        tk.Button(bar, text="Browse…", command=self.change_dir).pack(side="left", padx=4)

        # Folder list
        self.listbox = tk.Listbox(self)
        self.listbox.pack(fill="both", expand=True, padx=8, pady=4)
        self.listbox.bind("<Double-Button-1>", lambda e: self.run_xmla())

        # Action buttons
        btns = tk.Frame(self)
        btns.pack(pady=8)
        self.run_btn = tk.Button(btns, text="Run xmla", command=self.run_xmla)
        self.run_btn.pack(side="left", padx=4)
        tk.Button(btns, text="Close", command=self.destroy).pack(side="left", padx=4)

        self.load_folders()

    def load_folders(self):
        self.listbox.delete(0, tk.END)
        try:
            folders = sorted(
                d for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            )
        except OSError as e:
            messagebox.showerror("Error", str(e))
            return
        for d in folders:
            self.listbox.insert(tk.END, d)

    def change_dir(self):
        chosen = filedialog.askdirectory(initialdir=self.root_dir)
        if chosen:
            self.root_dir = chosen
            self.path_var.set(chosen)
            self.load_folders()

    def run_xmla(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("No selection", "Pick a folder first.")
            return
        folder = os.path.join(self.root_dir, self.listbox.get(sel[0]))
        self._set_busy(True)
        threading.Thread(target=self._worker, args=(folder,), daemon=True).start()

    def _worker(self, folder):
        try:
            xmla(folder)
        except Exception as e:
            msg = str(e)
            self.after(0, lambda m=msg: messagebox.showerror("xmla error", m))
        finally:
            self.after(0, lambda: self._set_busy(False))

    def _set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.run_btn.config(state=state)


if __name__ == "__main__":
    FolderPicker(root_dir=".").mainloop()