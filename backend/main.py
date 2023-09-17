from typing import Annotated
import uvicorn

from fastapi import BackgroundTasks, FastAPI, Request, Response, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from make_dataset_job import MakeDatasetJob
from train_job import TrainJob
from test_module import test_output
from io_module import get_file_name_list
from static_value import *

app = FastAPI()
templates = Jinja2Templates(directory="templates")
make_dataset_task = MakeDatasetJob()
train_task = TrainJob()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
  return templates.TemplateResponse("index.html", {"request": request, "file_name_list": get_file_name_list(MUSIC_ROOT + TRAIN_DIR)})

@app.get("/"+MUSIC_ROOT+TRAIN_DIR+"{file_name}")
async def get_train_music(file_name: str):
  with open(MUSIC_ROOT+TRAIN_DIR+file_name, "rb") as f:
    return Response(content=f.read(), media_type="audio/wav")

@app.get("/"+MUSIC_ROOT+TEST_DIR+"{file_name}")
async def get_train_music(file_name: str):
  with open(MUSIC_ROOT+TEST_DIR+file_name, "rb") as f:
    return Response(content=f.read(), media_type="audio/wav")

@app.post("/answer")
async def make_dataset(bg_tasks: BackgroundTasks, train_0: Annotated[int, Form()], train_1: Annotated[int, Form()], train_2: Annotated[int, Form()], train_3: Annotated[int, Form()], train_4: Annotated[int, Form()], train_5: Annotated[int, Form()], train_6: Annotated[int, Form()], train_7: Annotated[int, Form()], train_8: Annotated[int, Form()], train_9: Annotated[int, Form()]):
  answer_list = [train_0, train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8, train_9]
  bg_tasks.add_task(make_dataset_task, get_file_name_list(MUSIC_ROOT + TRAIN_DIR), answer_list)
  return RedirectResponse("/load/dataset", status_code=303)

@app.get("/load/dataset")
async def wait_dataset(bg_tasks: BackgroundTasks, request: Request):
  if(make_dataset_task.get_status()):
    now_file_name, now_progress = make_dataset_task.get_progress()
    return templates.TemplateResponse("load.html", {"request": request, "now_file_name": now_file_name, "now_progress": now_progress})
  else:
    train_data_dir, target_data_dir = make_dataset_task.get_dataset_dir()
    bg_tasks.add_task(train_task, train_data_dir, target_data_dir)
    return RedirectResponse("/train")

@app.get("/train")
async def train(request: Request):
  if(train_task.get_status()):
    now_epoch = train_task.get_progress()
    return templates.TemplateResponse("train.html", {"request": request, "now_epoch": now_epoch})
  else:
    return RedirectResponse("/test")

@app.get("/test")
async def test(request: Request):
  return templates.TemplateResponse("test.html", {"request": request, "file_name_list": get_file_name_list(MUSIC_ROOT + TEST_DIR), "file_rank_list": test_output(train_task.get_model_dir())})

if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)