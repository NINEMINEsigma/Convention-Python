import sys
import os
from time import sleep
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Convention.Runtime.Interaction import *

download = Interaction("http://www.liubai.site:4000/d/storage/Convention/Convention-Unity-Demo/TEST/song.mp3?sign=pIIWFqeous--i4H5fNIpqQsS0HeMinSRA_bztq6Czgo=:0")
file = download.Download("./[Test]/test.mp3")
sleep(5)
file.Remove()

