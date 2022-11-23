from fastapi import Form, File, UploadFile, Request, FastAPI
from typing import List
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from test_txtfolder import test
import json


app = FastAPI()


@app.post("/predict")
def check_result():
    result = test()
    #print(result)
    return {"JSON list" : str(result)}


if __name__ == "__main__":
    uvicorn.run(app)