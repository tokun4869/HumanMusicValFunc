import uvicorn
import time

from typing import Annotated
from fastapi import BackgroundTasks, FastAPI, Request, Response, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from module.train_job import TrainJob
from module.io import get_file_name_list, load_sound_list, save_user_data
from module.operation import test_operation
from module.const import *

app = FastAPI()
templates = Jinja2Templates(directory="templates")
train_task = TrainJob()
input_name_list = get_file_name_list(f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE_TRAIN}_{LISTEN_KEY}")
test_name_list = get_file_name_list(f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE_TEST}_{LISTEN_KEY}")
retest_name_list = get_file_name_list(f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE_RETEST}_{LISTEN_KEY}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
  if train_task.get_status() == STATUS_INPROGRESS:
    return RedirectResponse("/train")
  train_task.initialize()
  return templates.TemplateResponse("index.html", {"request": request, "file_name_list": input_name_list})

@app.get(f"/{MUSIC_ROOT}/{DATASET_TYPE}/"+"{mode_dir}/{file_name}")
async def get_music(mode_dir: str, file_name: str):
  with open(f"{MUSIC_ROOT}/{DATASET_TYPE}/{mode_dir}/{file_name}", "rb") as f:
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
    return templates.TemplateResponse("train.html", {"request": request, "now_epoch": now_epoch, "num_epochs": NUM_EPOCHS})
  elif(status == STATUS_BEFORE):
    time.sleep(1)
    return RedirectResponse("/train")
  else:
    return {"Status": train_task.get_error()}

@app.get("/test")
async def test(request: Request):
  file_name_list = get_file_name_list(f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE_TEST}_{LISTEN_KEY}")
  sound_list = load_sound_list(file_name_list)
  return templates.TemplateResponse("test.html", {"request": request, "file_name_list": test_name_list, "file_rank_list": test_operation(train_task.get_model_dir(), sound_list)})

@app.get("/{dataset}", response_class=HTMLResponse)
async def user_input_page(request: Request, dataset: str):
  return templates.TemplateResponse("user.html", {"request": request, "dataset": dataset, "answered_url": f"{dataset}/answer", "train_file_name_list": input_name_list, "test_file_name_list": test_name_list, "retest_file_name_list": retest_name_list})

@app.post("/{dataset}/answer/")
async def save_answer_data(
  dataset: str, name: Annotated[str, Form()],
  train_0: Annotated[int, Form()], train_1: Annotated[int, Form()], train_2: Annotated[int, Form()], train_3: Annotated[int, Form()], train_4: Annotated[int, Form()], train_5: Annotated[int, Form()], train_6: Annotated[int, Form()], train_7: Annotated[int, Form()], train_8: Annotated[int, Form()], train_9: Annotated[int, Form()],
  test_0: Annotated[int, Form()], test_1: Annotated[int, Form()], test_2: Annotated[int, Form()], test_3: Annotated[int, Form()], test_4: Annotated[int, Form()], test_5: Annotated[int, Form()], test_6: Annotated[int, Form()], test_7: Annotated[int, Form()], test_8: Annotated[int, Form()], test_9: Annotated[int, Form()],
  retest_0: Annotated[int, Form()], retest_1: Annotated[int, Form()], retest_2: Annotated[int, Form()], retest_3: Annotated[int, Form()], retest_4: Annotated[int, Form()], retest_5: Annotated[int, Form()], retest_6: Annotated[int, Form()], retest_7: Annotated[int, Form()], retest_8: Annotated[int, Form()], retest_9: Annotated[int, Form()]):
  train_list = [train_0, train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8, train_9]
  test_list = [test_0, test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9]
  retest_list = [retest_0, retest_1, retest_2, retest_3, retest_4, retest_5, retest_6, retest_7, retest_8, retest_9]
  save_user_data(dataset, name, train_list, test_list, retest_list)
  return RedirectResponse(f"/{dataset}/answer/finish", status_code=303)

@app.get("/{dataset}/answer/finish", response_class=HTMLResponse)
async def answer_finish(request: Request, dataset: str):
  return templates.TemplateResponse("userfin.html", {"request": request, "dataset": dataset})

@app.get("/{dataset}"+f"/{MUSIC_ROOT}/{DATASET_TYPE}/"+"{mode_dir}/{file_name}")
async def get_user_music(mode_dir: str, file_name: str):
  with open(f"{MUSIC_ROOT}/{DATASET_TYPE}/{mode_dir}/{file_name}", "rb") as f:
    return Response(content=f.read(), media_type="audio/mp3")

if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)