# ------------------ gui_app.py ------------------

import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("parkinson_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names
features = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
    "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

# ------------------ GUI ------------------
root = tk.Tk()
root.title("Parkinson's Disease Detection (RandomForest)")
root.geometry("650x900")
root.resizable(False, False)

tk.Label(root, text="Parkinson's Disease Prediction System",
         font=("Arial", 16, "bold")).pack(pady=10)

entries = []
frame = tk.Frame(root)
frame.pack()

# Create input fields
for f in features:
    row = tk.Frame(frame)
    lbl = tk.Label(row, width=22, text=f + ":", anchor='w')
    ent = tk.Entry(row, width=20)

    row.pack(side=tk.TOP, fill=tk.X, pady=3)
    lbl.pack(side=tk.LEFT)
    ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)

    entries.append(ent)

# Prediction function
def predict():
    try:
        values = [float(entry.get()) for entry in entries]
        arr = np.array([values])
        scaled = scaler.transform(arr)
        result = model.predict(scaled)[0]

        if result == 1:
            messagebox.showerror("Prediction Result",
                                 "🔴 The person is LIKELY to have Parkinson's Disease.")
        else:
            messagebox.showinfo("Prediction Result",
                                "🟢 The person is HEALTHY.")
    
    except ValueError:
        messagebox.showwarning("Input Error",
                               "Please enter all values correctly (numbers only).")

# Predict Button
tk.Button(root, text="Predict", font=("Arial", 14), bg="green",
          fg="white", command=predict).pack(pady=20)

# Show Feature Importance Button
def show_graph():
    import os
    if os.path.exists("feature_importance.png"):
        os.startfile("feature_importance.png")
    else:
        messagebox.showwarning("Error", "Graph file not found!")

tk.Button(root, text="Show Feature Importance Graph",
          font=("Arial", 12), bg="blue", fg="white",
          command=show_graph).pack(pady=5)

root.mainloop()