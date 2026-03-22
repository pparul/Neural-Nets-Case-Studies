import gradio as gr
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model_definition import Mymodel

# Load model (same as before)
model = Mymodel(din=3, dout=4)
model.load_state_dict(torch.load("model.pt"))
model.eval()

def predict(f1, f2, f3):
    x = torch.tensor([[f1, f2, f3]], dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).item()

    # Create a plot
    fig, ax = plt.subplots()
    ax.bar(["Feature 1", "Feature 2", "Feature 3"], [f1, f2, f3])
    ax.set_title(f"Prediction: {pred:.4f}")
    return pred, fig


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(-3, 3, label="Feature 1"),
        gr.Slider(-3, 3, label="Feature 2"),
        gr.Slider(-3, 3, label="Feature 3"),
    ],
    outputs=[
        gr.Number(label="Prediction"),
        gr.Plot(label="Feature Plot"),
    ],
    title="My Neural Network",
)

demo.launch()
