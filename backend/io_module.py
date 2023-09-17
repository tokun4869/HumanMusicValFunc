import glob

def get_file_name_list(dir: str):
  return glob.glob(dir + "/*.wav")

def get_new_file_path(dir: str, base: str, ext: str):
  file_name_list = glob.glob("{}/{}_*{}".format(dir, base, ext))
  max_index = -1
  for file_name in file_name_list:
    index = int(file_name[len(dir)+len("/")+len(base)+len("_") : -len(ext)])
    if max_index < index:
      max_index = index
  path = "{}/{}_{}{}".format(dir, base, max_index+1, ext)
  return path