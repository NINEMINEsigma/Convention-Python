import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Convention.Runtime.Config import *
from Convention.Runtime.EasySave import *

class Test:
    test_field:int = 10
    class_test_field:int = 20

    def __init__(self):
        self.test_field:int = 0

def run():
    print(Test.__annotations__)

if __name__ == "__main__":
    run()
