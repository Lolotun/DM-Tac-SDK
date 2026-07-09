import robotiq_gripper
import time
ip = "192.168.88.56"
gripper = robotiq_gripper.RobotiqGripper()
gripper.connect(ip, 63352)
# print(gripper.get_status())
 pos =gripper.move(200, 0, 0)

# pos , status = gripper.move_and_wait_for_pos(200, 0, 0)
# gripper.move(100, 0, 0)
print( f"pos:{pos} , status: {status}")
# now = time.time()