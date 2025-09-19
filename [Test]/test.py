import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Convention.Runtime.File import *

first = ToolFile("E:/dev/")
second = ToolFile("/analyze/")
print(first|second)