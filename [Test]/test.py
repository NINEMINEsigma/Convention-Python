import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Convention.Runtime.Config import *
from Convention.Runtime.EasySave import *

class test_log(BaseModel):
    test_field:int
    test_field_2:str

def run():
    SetInternalDebug(True)
    SetInternalReflectionDebug(True)
    SetInternalEasySaveDebug(True)
    test = test_log(test_field=1,test_field_2="test")
    EasySave.Write(test,"test.json")

if __name__ == "__main__":
    run()
