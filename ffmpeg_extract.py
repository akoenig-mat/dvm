import subprocess

def extractf( movie_name, folder ):
    subprocess.call(
    "ffmpeg -i " +str(movie_name) +" -vf scale=256:256 "+str(folder)+"frames_%03d.png", shell=True)
    print( str(movie_name)+" extracted")
    return

