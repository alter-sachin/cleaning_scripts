from moviepy.editor import VideoFileClip, concatenate_videoclips
import os 
import shutil


y = os.listdir()
clip1 = VideoFileClip("00M29S_1555601429.mp4")
for clip in y:
	print(clip)
	if ".mp4" in clip:
		clipx = VideoFileClip(clip)
		clip1 = concatenate_videoclips([clip1,clipx])


clip1.write_videofile("my_concatenation.mp4")




#clip1 = VideoFileClip("myvideo.mp4")
#clip2 = VideoFileClip("myvideo2.mp4").subclip(50,60)
#clip3 = VideoFileClip("myvideo3.mp4")
#final_clip = concatenate_videoclips([clip1,clip2,clip3])

