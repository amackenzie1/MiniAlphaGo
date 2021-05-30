import sys 
import os

num_processes = int(sys.argv[1])
episode_length = int(sys.argv[2])

command = ""

for i in range(num_processes):
    command += f"python mcts_cpu.py {episode_length} & "
command += "ls"

os.system(command)