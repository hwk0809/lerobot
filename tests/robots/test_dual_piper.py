from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lerobot.robots.dual_piper import DualPiper, DualPiperConfig
from lerobot.robots.dual_piper.dual_piper import ARM_FACTOR


class FakePiperInterface:
    def __init__(self, port: str, *, joint_scale: float, gripper_raw: int):
        self.port = port
        self.joint_scale = joint_scale
        self.gripper_raw = gripper_raw
        self.connected = False

    def ConnectPort(self) -> None:
        self.connected = True

    def DisconnectPort(self) -> None:
        self.connected = False

    def GetArmJointMsgs(self):
        joints = {
            f"joint_{i}": round(i * self.joint_scale * ARM_FACTOR)
            for i in range(1, 7)
        }
        return SimpleNamespace(joint_state=SimpleNamespace(**joints))

    def GetArmGripperMsgs(self):
        state = SimpleNamespace(grippers_angle=self.gripper_raw)
        return SimpleNamespace(gripper_state=state)


@pytest.fixture
def robot(tmp_path):
    interfaces = {
        "left_bus": FakePiperInterface("left_bus", joint_scale=0.1, gripper_raw=35_000),
        "right_bus": FakePiperInterface("right_bus", joint_scale=-0.1, gripper_raw=70_000),
    }
    config = DualPiperConfig(
        id="test_dual_piper",
        calibration_dir=tmp_path,
        left_port="left_bus",
        right_port="right_bus",
    )
    with patch(
        "lerobot.robots.dual_piper.dual_piper._make_piper_interface",
        side_effect=lambda port: interfaces[port],
    ):
        instance = DualPiper(config)
    return instance, interfaces


def test_two_distinct_can_ports_are_required():
    with pytest.raises(ValueError, match="two different CAN buses"):
        DualPiperConfig(left_port="can0", right_port="can0")


def test_reads_normalized_follower_state_from_two_buses(robot):
    instance, interfaces = robot
    instance.connect()

    state = instance.get_joint_state()

    assert interfaces["left_bus"].connected
    assert interfaces["right_bus"].connected
    assert state["left_joint_1.pos"] == pytest.approx(0.1, abs=2e-5)
    assert state["left_joint_6.pos"] == pytest.approx(0.6, abs=2e-5)
    assert state["left_gripper.pos"] == 0.5
    assert state["right_joint_1.pos"] == pytest.approx(-0.1, abs=2e-5)
    assert state["right_joint_6.pos"] == pytest.approx(-0.6, abs=2e-5)
    assert state["right_gripper.pos"] == 1.0

    # Firmware linkage drives the followers; this adapter must remain passive.
    assert instance.send_action({"left_joint_1.pos": 99.0}) == state

    instance.disconnect()
    assert not interfaces["left_bus"].connected
    assert not interfaces["right_bus"].connected
    assert not instance.is_connected
