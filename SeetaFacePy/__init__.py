import sys,os
file_dir = os.path.dirname(__file__)

print ("Python Version {}".format(sys.version))
platform = sys.platform
if platform == "win32":
    lib_dir = os.path.join(file_dir,"lib","win")
elif platform == "linux":
    lib_dir = os.path.join(file_dir,"lib","linux")
else:
    print("Unsupported platform:{}".format(platform))
    exit()
os.environ["PATH"]+=os.pathsep+lib_dir
try:
    from  .api import *
except Exception as e :
    print(e)
