from collections.abc import Iterable
from flygym import assets_dir
from flygym.compose import Fly, ActuatorType, KinematicPose
from flygym.anatomy import (
    AxisOrder,
    ActuatedDOFPreset,
    JointPreset,
    Skeleton,
    BodySegment,
)

LEG_NAMES = ["lf", "lm", "lh", "rf", "rm", "rh"]


def create_fly(
    adhesion_gain: float = 50,
    position_gain: float = 50,
    adhesion_segments: Iterable[str] | None = tuple(
        f"{leg}_tarsus5" for leg in LEG_NAMES
    ),
    axis_order: AxisOrder = AxisOrder.YAW_PITCH_ROLL,
    joint_preset: JointPreset = JointPreset.LEGS_ONLY,
    dof_preset: ActuatedDOFPreset = ActuatedDOFPreset.LEGS_ACTIVE_ONLY,
    actuator_type: ActuatorType = ActuatorType.POSITION,
    neutral_pose_path=assets_dir / "model/pose/neutral.yaml",
    **kwargs,
):
    fly = Fly(**kwargs)
    skeleton = Skeleton(axis_order=axis_order, joint_preset=joint_preset)
    neutral_pose = KinematicPose(path=neutral_pose_path)
    fly.add_joints(skeleton, neutral_pose=neutral_pose)

    actuated_dofs = fly.skeleton.get_actuated_dofs_from_preset(dof_preset)
    fly.add_actuators(
        actuated_dofs,
        actuator_type=actuator_type,
        kp=position_gain,
        neutral_input=neutral_pose,
    )
    fly.add_odor_sensors()
    fly.add_vision()

    adhesion_segments = [
        seg if isinstance(seg, BodySegment) else BodySegment(seg)
        for seg in adhesion_segments
    ]
    fly.add_adhesion_actuators(segments=adhesion_segments, gain=adhesion_gain)
    fly.add_force_sensors()
    fly.add_antenna_joints()
    fly.colorize()
    return fly
