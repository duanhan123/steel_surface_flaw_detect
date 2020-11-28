from test import *
from CNNsDemo import *
import time

def main():
    startTime = time.time()
    # train()
    endTime = time.time()
    print("train time: %d s" % (endTime - startTime))
    for i in range(10):
        acc_cal(100)
    display(20)

if __name__ == "__main__":
    # torch.backends.cudnn.enabled = False
    main()
