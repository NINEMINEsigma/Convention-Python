import sys
import os
from time import sleep
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Convention.Runtime.File import *

file = ToolFile("[Test]")|"temp"|None
print(file.MustExistsPath())

