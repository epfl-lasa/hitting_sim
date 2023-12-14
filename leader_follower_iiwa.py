import numpy as np
import os
import torch

import pybullet as p
import pybullet_data

from ds import linear_ds
from controller import get_joint_velocities_qp
from get_dual_system import sim_dual_robot


#############################################################
system = sim_dual_robot()


