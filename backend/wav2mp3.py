import glob
import torchaudio

if __name__ == "__main__":
  wav_file_dir = "data/music/wav/"
  mp3_file_dir = "data/music/mp3/"
  wav_file_name_list = glob.glob(wav_file_dir + "*.wav")
  print("=== start wav to mp3 ===")
  for index, wav_file_name in enumerate(wav_file_name_list):
    print(f"{wav_file_name} [{index+1}/{len(wav_file_name_list)}]")
    wav_file, sample_rate = torchaudio.load(wav_file_name)
    sample_length = sample_rate*180
    num_slice = int(wav_file.size()[1]/sample_length)
    if num_slice == 0:
      print("slice num [1/1]")
      mp3_file_name = mp3_file_dir + wav_file_name[len(wav_file_dir):-len(".wav")] + "_0.mp3"
      torchaudio.save(uri=mp3_file_name, src=src_wav, sample_rate=sample_rate, format="mp3")
    else:
      for slice_index in range(int(wav_file.size()[1]/sample_length)):
        print(f"slice num [{slice_index+1}/{num_slice}]")
        l = sample_length*slice_index
        r = sample_length*(slice_index+1) if sample_length*(slice_index+1) < wav_file.size()[1] else wav_file.size()[1]
        src_wav = wav_file[:, l:r]
        mp3_file_name = mp3_file_dir + wav_file_name[len(wav_file_dir):-len(".wav")] + f"_{slice_index}.mp3"
        torchaudio.save(uri=mp3_file_name, src=src_wav, sample_rate=sample_rate, format="mp3")
  print("finish!")