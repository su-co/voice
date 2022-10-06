import pytube
import os

url = "https://www.youtube.com/watch?v=FM7MFYoylVs&ab_channel=ChainsmokersVEVO"
output_path = os.path.join('..', 'data')
file_name = "something just like this"

pytube.YouTube(url).streams.filter(  # 下载文件
    only_audio=True, file_extension="mp4")[0].download(output_path=output_path, filename=file_name)

