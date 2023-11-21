import sys
from static_value import *
from test_module import test_output

if __name__ == "__main__":
  try:
    model_num = sys.argv[1]
    ext = ".pth"
    model_path = MODEL_ROOT + str(model_num) + ext
    test_output(model_path)
  except Exception as e:
    print(str(e))