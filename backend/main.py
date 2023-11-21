import uvicorn
import time

from typing import Annotated
from fastapi import BackgroundTasks, FastAPI, Request, Response, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from train_job import TrainJob
from test_module import test_output
from io_module import get_file_name_list, save_user_data
from static_value import *

app = FastAPI()
templates = Jinja2Templates(directory="templates")
train_task = TrainJob()
input_name_list = get_file_name_list(MUSIC_ROOT + TRAIN_LISTEN_DIR)
test_name_list = get_file_name_list(MUSIC_ROOT + TEST_LISTEN_DIR)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
  if train_task.get_status() == STATUS_INPROGRESS:
    return RedirectResponse("/train")
  train_task.initialize()
  return templates.TemplateResponse("index.html", {"request": request, "file_name_list": input_name_list})

@app.get("/"+MUSIC_ROOT+TRAIN_LISTEN_DIR+"{file_name}")
async def get_train_music(file_name: str):
  with open(MUSIC_ROOT+TRAIN_LISTEN_DIR+file_name, "rb") as f:
    return Response(content=f.read(), media_type="audio/mp3")

@app.get("/"+MUSIC_ROOT+TEST_LISTEN_DIR+"{file_name}")
async def get_test_music(file_name: str):
  with open(MUSIC_ROOT+TEST_LISTEN_DIR+file_name, "rb") as f:
    return Response(content=f.read(), media_type="audio/mp3")

@app.post("/answer")
async def make_dataset(bg_tasks: BackgroundTasks, train_0: Annotated[int, Form()], train_1: Annotated[int, Form()], train_2: Annotated[int, Form()], train_3: Annotated[int, Form()], train_4: Annotated[int, Form()], train_5: Annotated[int, Form()], train_6: Annotated[int, Form()], train_7: Annotated[int, Form()], train_8: Annotated[int, Form()], train_9: Annotated[int, Form()]):
  rank_list = [train_0, train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8, train_9]
  bg_tasks.add_task(train_task, rank_list)
  return RedirectResponse("/train", status_code=303)

@app.get("/train")
async def train(request: Request):
  status = train_task.get_status()
  if(status == STATUS_FINISH):
    return RedirectResponse("/test")
  elif(status == STATUS_INPROGRESS):
    now_epoch = train_task.get_now_epoch()
    num_epochs = train_task.get_num_epochs()
    return templates.TemplateResponse("train.html", {"request": request, "now_epoch": now_epoch, "num_epochs": num_epochs})
  elif(status == STATUS_BEFORE):
    time.sleep(1)
    return RedirectResponse("/train")
  else:
    return {"Status": train_task.get_error()}

@app.get("/test")
async def test(request: Request):
  return templates.TemplateResponse("test.html", {"request": request, "file_name_list": test_name_list, "file_rank_list": test_output(train_task.get_model_dir())})

@app.get("/user/", response_class=HTMLResponse)
async def user_input_page(request: Request):
  return templates.TemplateResponse("user.html", {"request": request, "train_file_name_list": input_name_list, "test_file_name_list": test_name_list})

@app.post("/user/answer")
async def save_answer_data(name: Annotated[str, Form()], train_0: Annotated[int, Form()], train_1: Annotated[int, Form()], train_2: Annotated[int, Form()], train_3: Annotated[int, Form()], train_4: Annotated[int, Form()], train_5: Annotated[int, Form()], train_6: Annotated[int, Form()], train_7: Annotated[int, Form()], train_8: Annotated[int, Form()], train_9: Annotated[int, Form()], test_0: Annotated[int, Form()], test_1: Annotated[int, Form()], test_2: Annotated[int, Form()], test_3: Annotated[int, Form()], test_4: Annotated[int, Form()], test_5: Annotated[int, Form()], test_6: Annotated[int, Form()], test_7: Annotated[int, Form()], test_8: Annotated[int, Form()], test_9: Annotated[int, Form()]):
  train_list = [train_0, train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8, train_9]
  test_list = [test_0, test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9]
  save_user_data(name, train_list, test_list)
  return RedirectResponse("/user/answer/finish", status_code=303)

@app.get("/user/answer/finish", response_class=HTMLResponse)
async def answer_finish(request: Request):
  return templates.TemplateResponse("userfin.html", {"request": request})

@app.get("/user/"+MUSIC_ROOT+TRAIN_LISTEN_DIR+"{file_name}")
async def get_user_train_music(file_name: str):
  with open(MUSIC_ROOT+TRAIN_LISTEN_DIR+file_name, "rb") as f:
    return Response(content=f.read(), media_type="audio/mp3")

@app.get("/user/"+MUSIC_ROOT+TEST_LISTEN_DIR+"{file_name}")
async def get_user_test_music(file_name: str):
  with open(MUSIC_ROOT+TEST_LISTEN_DIR+file_name, "rb") as f:
    return Response(content=f.read(), media_type="audio/mp3")

if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)