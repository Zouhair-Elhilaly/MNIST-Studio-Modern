import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
# Make sure the following files exist in your project
from predictor import Predictor
from pretraitement import preprocess_image

class MNISTStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Studio - Modern Edition")
        self.root.geometry("700x450")
        self.root.configure(bg="#f0f2f5")  # Calm light gray background

        # Colors and fonts
        self.primary_color = "#4a90e2"
        self.secondary_color = "#2c3e50"
        self.accent_color = "#ffffff"
        self.font_main = ("Segoe UI", 12)
        self.font_header = ("Segoe UI", 18, "bold")

        # Predictor
        self.predictor = Predictor("../models/best_model_v2.pth")

        # --- Divide UI into frames ---
        self.left_frame = tk.Frame(root, bg="#f0f2f5", padx=20, pady=20)
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(root, bg="white", padx=20, pady=20, highlightbackground="#d1d9e0", highlightthickness=1)
        self.right_frame.pack(side="right", fill="both", expand=True)

        # --- Left side: Drawing area ---
        self.canvas_width = 300
        self.canvas_height = 300
        
        # Add subtle shadow around canvas
        self.canvas_container = tk.Frame(self.left_frame, bg="#bdc3c7", padx=2, pady=2)
        self.canvas_container.pack()

        self.canvas = tk.Canvas(self.canvas_container, width=self.canvas_width, height=self.canvas_height, 
                                bg="white", cursor="cross", highlightthickness=0)
        self.canvas.pack()

        # Control buttons
        self.btn_frame = tk.Frame(self.left_frame, bg="#f0f2f5")
        self.btn_frame.pack(fill="x", pady=15)

        self.btn_predict = tk.Button(self.btn_frame, text="Predict Number", command=self.predict, 
                                     bg=self.primary_color, fg="white", font=self.font_main, 
                                     relief="flat", height=1, width=12, cursor="hand2")
        self.btn_predict.pack(side="left", padx=5, expand=True)

        self.btn_clear = tk.Button(self.btn_frame, text="Clear Canvas", command=self.clear_canvas, 
                                   bg="#e74c3c", fg="white", font=self.font_main, 
                                   relief="flat", height=1, width=12, cursor="hand2")
        self.btn_clear.pack(side="right", padx=5, expand=True)

        # --- Right side: Results ---
        tk.Label(self.right_frame, text="Smart Analysis", font=self.font_header, bg="white", fg=self.secondary_color).pack(pady=(0, 20))

        self.result_container = tk.Frame(self.right_frame, bg="#f8f9fa", pady=20)
        self.result_container.pack(fill="x")

        self.result_label = tk.Label(self.result_container, text="Result: --", font=("Segoe UI", 20), bg="#f8f9fa", fg=self.primary_color)
        self.result_label.pack()

        self.conf_label = tk.Label(self.right_frame, text="Confidence: 0%", font=self.font_main, bg="white", fg="#7f8c8d")
        self.conf_label.pack(pady=10)

        # Visual progress bar (optional for aesthetics)
        self.progress_bg = tk.Frame(self.right_frame, bg="#ecf0f1", height=10, width=200)
        self.progress_bg.pack(pady=5)
        self.progress_bar = tk.Frame(self.progress_bg, bg=self.primary_color, height=10, width=0)
        self.progress_bar.place(x=0, y=0)

        # Drawing setup
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        # Slightly thicker line to match MNIST style
        r = 8 
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=255)
        self.result_label.config(text="Result: --")
        self.conf_label.config(text="Confidence: 0%")
        self.update_progress_bar(0)

    def update_progress_bar(self, confidence):
        # Update confidence bar visually
        width = int(200 * confidence)
        self.progress_bar.config(width=width)

    def predict(self):
        img_array = np.array(self.image1)
        processed_img = preprocess_image(img_array)
        res, conf = self.predictor.predict(processed_img)
        if conf < 0.9:
            messagebox.showwarning("Low Confidence", "The model is not confident about this prediction. Please try again.")
        self.result_label.config(text=f"Result: {res}")
        self.conf_label.config(text=f"Confidence: {conf*100:.1f}%")
        self.update_progress_bar(conf)
        
        # Change result color based on confidence
        if conf >= 0.90:
            self.result_label.config(fg="#27ae60")
        elif conf >= 0.70:
            self.result_label.config(fg="#f39c12")
        else:
            self.result_label.config(fg="#e74c3c")

if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTStudioApp(root)
    root.mainloop()