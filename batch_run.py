import os
import random

num_classes = 10
trigger_names = ['Trigger2','solid','square']

for t in trigger_names:
  for sid in range(num_classes):
    tid = sid
    while tid == sid:
      tid = random.randrange(num_classes)

    c1 = 2
    c2 = 3
    uu = [0]*num_classes
    uu[sid] = 1
    uu[tid] = 1
    for i in range(2,num_classes):
      if uu[i]==0:
        c1 = i
        break
    uu[c1] = 1
    for i in range(3,num_classes):
      if uu[i]==0:
        c2 = i
        break
    uu[c2] = 1

    cmmd = 'python train_cifar10_keras.py %d %d %d %d '%(sid,tid,c1,c2)
    cmmd = cmmd+t
    print(cmmd)
    os.system(cmmd)


    log_name = 'cifar10_s%d_t%d_c%d%d_'%(sid,tid,c1,c2)
    log_name = log_name+t+'.txt'
    cmmd = 'python run_abs.py 1>logs/'+log_name
    print(cmmd)
    os.system(cmmd)

