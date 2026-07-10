import robotiq_gripper
import time
import threading

ip = "192.168.88.56"
gripper = robotiq_gripper.RobotiqGripper()
gripper.connect(ip, 63352)
PATH = "/home/physicalai/Denmark/DM-Tac-SDK/proba.txt"
# gripper.move(200, 0, 0)

# # pos , status = gripper.move_and_wait_for_pos(200, 0, 0)
# # gripper.move(100, 0, 0)
# print( f"pos:{pos} , status: {status}")
# # now = time.time()
lock = threading.Lock()
running = True
target = gripper.get_current_position()

def keyboard_listener():
    global running, target
    while running:
        key = input("Print cmd from 0 to 240:  ")
        with lock:
            if key.lower() == 'q':
                running = False
                break
            try:
                tgt = int(key)
                if 0 <= tgt <= 255:
                    target = tgt
                else:
                    print("Please enter a value between 0 and 255.")
                    print()
            except ValueError:
                print("Invalid input. Please enter a number or 'q' to quit.")

        
thread = threading.Thread(target=keyboard_listener)
thread.start()
PERIOD = 0.1  
next_t = time.perf_counter()
with open(PATH, 'w', encoding='utf-8') as f:  
    while running:
        with lock:
            tgt = target
        gripper.move(tgt, speed=0, force=0)
        pos = gripper.get_current_position()

        f.write(f"target={tgt} pos={pos} time ={next_t}\n")    
        f.flush()                                

        next_t += PERIOD
        sleep = next_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)

gripper.disconnect()