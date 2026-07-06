import numpy as np
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = x + y
    
    return {"message": "Hello World!", "addition": z.tolist(), "product": int(x@y)}