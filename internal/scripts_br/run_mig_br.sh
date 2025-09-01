#!/bin/bash


# profiles are specified to each GPU, e.g. profile 15 can be used to divide into 4 devices of size 20gb
# NVIDIA H100 80GB HBM3
#| => sudo nvidia-smi mig -i 3 -cgi 15 -C
#Successfully created GPU instance ID  5 on GPU  3 using profile MIG 1g.20gb (ID 15)
#Successfully created compute instance ID  0 on GPU  3 GPU instance ID  5 using profile MIG 1g.20gb (ID  7)




# GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-afddd1b4-4464-96c8-a712-aaeb0acf1170)  # cudo 0 on torch
# GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-6faf0136-7870-5767-10be-a0827a158829)
# GPU 2: NVIDIA H100 80GB HBM3 (UUID: GPU-20d20fc3-bcc7-e715-32d6-ffd646ea062f)
# GPU 3: NVIDIA H100 80GB HBM3 (UUID: GPU-182e6bd5-b7ac-e0a6-48cf-96e198063dd3)
#   MIG 1g.20gb     Device  0: (UUID: MIG-56679450-0984-50db-83a3-7e549eb60883)  # cudo 4 on torch
#   MIG 1g.20gb     Device  1: (UUID: MIG-a155b8d5-2484-52fc-a2ed-e47dc89996cd)
#   MIG 1g.20gb     Device  2: (UUID: MIG-9dc27b3c-b567-5802-a2a7-27ad657ab079)
#   MIG 1g.20gb     Device  3: (UUID: MIG-f6102e7f-bbf5-5db4-abea-156619dd4ce2)



# Split into to 40gb device sudo nvidia-smi mig -i 5 -cgi 5,5

# (0) choose a device
DEVICE_INDEX_FOR_MIG=1
PROFILE=15
PROFILE=9 # 

# (1) show all gpu indices, uuids,  and product names
nvidia-smi -L

# (2) list all MIG instances
sudo nvidia-smi mig -lgi

# (3) activate multi-instance gpu for 
sudo nvidia-smi --id ${DEVICE_INDEX_FOR_MIG} -mig 1

# split device with index 0 into 3 compute instances
for i in {0..3}; do
    sudo nvidia-smi mig --id ${DEVICE_INDEX_FOR_MIG} -cgi ${PROFILE} -C
done

# show all gpu indices, uuids, and produce names
nvidia-smi -L
