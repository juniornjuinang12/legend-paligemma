import os
import sys
import time
import socket
import platform

print("=== SALAD LOG TEST: START ===", flush=True)
print("python:", sys.version.replace("\n", " "), flush=True)
print("platform:", platform.platform(), flush=True)
print("hostname:", socket.gethostname(), flush=True)
print("cwd:", os.getcwd(), flush=True)
print("files:", os.listdir("."), flush=True)
print("HF_TOKEN present:", bool(os.getenv("HF_TOKEN")), flush=True)
print("=== SALAD LOG TEST: END (looping) ===", flush=True)

i = 0
while True:
    i += 1
    print(f"tick={i} time={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    time.sleep(5)
