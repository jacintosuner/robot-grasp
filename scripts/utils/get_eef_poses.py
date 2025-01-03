from frankapy import FrankaArm
from autolab_core import RigidTransform
from frankapy.utils import franka_pose_to_rigid_transform

print('Starting robot')
fa = FrankaArm()
print(fa.get_pose())


#fa.set_tool_delta_pose

print(fa.get_pose(include_tool_offset=False))
print(fa._tool_delta_pose)