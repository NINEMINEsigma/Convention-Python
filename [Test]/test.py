import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Convention.Runtime.Config import *

def run():
    print_colorful(ConsoleFrontColor.RED,"test")

if __name__ == "__main__":
    run()
