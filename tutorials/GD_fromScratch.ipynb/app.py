from fastapi import FastAPI
import torch
import torch.nn as nn

class Mymodel(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.ff1 = nn.Linear(din, dout)
        self.ff2 = nn.Linear(dout, 1)

    def forward(self, x):
        x = self.ff1(x)
        x = torch.sigmoid(x)
        x = self.ff2(x)
        x = torch.sigmoid(x)
        return x

# Load model
model = Mymodel(din=3, dout=4)
model.load_state_dict(torch.load("model.pt"))
model.eval()

app = FastAPI()

from pydantic import BaseModel

class PredictRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(request: PredictRequest):
    x = torch.tensor([request.features], dtype=torch.float32)
    with torch.no_grad():
        pred = model(x)
    return {"prediction": pred.item()}

