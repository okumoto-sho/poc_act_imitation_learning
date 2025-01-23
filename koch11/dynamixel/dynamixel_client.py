import dynamixel_sdk.robotis_def as robotis_def
from collections import namedtuple
from dynamixel_sdk import (
    PortHandler,
    PacketHandler,
    GroupSyncRead,
    GroupSyncWrite,
    DXL_LOBYTE,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOWORD,
)
from enum import Enum
from typing import List

ControlTableElementDescription = namedtuple(
    "ControlTableElement", ["address", "num_bytes", "is_read_only"]
)


class ControlTable(Enum):
    ModelNumber = ControlTableElementDescription(0, 2, True)
    ModelInformation = ControlTableElementDescription(2, 4, True)
    FirmwareVersion = ControlTableElementDescription(6, 1, True)
    Id = ControlTableElementDescription(7, 1, False)
    BaudRate = ControlTableElementDescription(8, 1, False)
    ReturnDelayTime = ControlTableElementDescription(9, 1, False)
    DriveMode = ControlTableElementDescription(10, 1, False)
    OperatingMode = ControlTableElementDescription(11, 1, False)
    SecondaryId = ControlTableElementDescription(12, 1, False)
    ProtocolType = ControlTableElementDescription(13, 1, False)
    HomingOffset = ControlTableElementDescription(20, 4, False)
    MovingThreshold = ControlTableElementDescription(24, 4, False)
    TemperatureLimit = ControlTableElementDescription(31, 1, False)
    MaxVoltageLimit = ControlTableElementDescription(32, 2, False)
    MinVoltageLimit = ControlTableElementDescription(34, 2, False)
    PwmLimit = ControlTableElementDescription(36, 2, False)
    VelocityLimit = ControlTableElementDescription(44, 4, False)
    MaxPositionLimit = ControlTableElementDescription(48, 4, False)
    MinPositionLimit = ControlTableElementDescription(52, 4, False)
    StartupConfiguration = ControlTableElementDescription(60, 1, False)
    Shutdown = ControlTableElementDescription(63, 1, False)
    TorqueEnable = ControlTableElementDescription(64, 1, False)
    Led = ControlTableElementDescription(65, 1, False)
    StatusReturnLevel = ControlTableElementDescription(68, 1, False)
    RegisteredInstruction = ControlTableElementDescription(69, 1, True)
    HardwareErrorStatus = ControlTableElementDescription(70, 1, True)
    VelocityIGain = ControlTableElementDescription(76, 2, False)
    VelocityPGain = ControlTableElementDescription(78, 2, False)
    PositionDGain = ControlTableElementDescription(80, 2, False)
    PositionIGain = ControlTableElementDescription(82, 2, False)
    PositionPGain = ControlTableElementDescription(84, 2, False)
    FeedForward2ndGain = ControlTableElementDescription(88, 2, False)
    FeedForward1stGain = ControlTableElementDescription(90, 2, False)
    BusWatchdog = ControlTableElementDescription(98, 1, False)
    GoalPwm = ControlTableElementDescription(100, 2, False)
    GoalVelocity = ControlTableElementDescription(104, 4, False)
    ProfileAcceleration = ControlTableElementDescription(108, 4, False)
    ProfileVelocity = ControlTableElementDescription(112, 4, False)
    GoalPosition = ControlTableElementDescription(116, 4, False)
    RealtimeTick = ControlTableElementDescription(120, 2, True)
    Moving = ControlTableElementDescription(122, 1, True)
    MovingStatus = ControlTableElementDescription(123, 1, True)
    PresentPwm = ControlTableElementDescription(124, 2, True)
    PresentLoad = ControlTableElementDescription(126, 2, True)
    PresentVelocity = ControlTableElementDescription(128, 4, True)
    PresentPosition = ControlTableElementDescription(132, 4, True)
    VelocityTrajectory = ControlTableElementDescription(136, 4, True)
    PositionTrajectory = ControlTableElementDescription(140, 4, True)
    PresentInputVoltage = ControlTableElementDescription(144, 2, True)
    PresentTemperature = ControlTableElementDescription(146, 1, True)
    BackupReady = ControlTableElementDescription(147, 1, True)


class DynamixelXLSeriesClient:
    def __init__(self, port_name="/dev/ttyACM0", baud_rate=2000000):
        self.port_handler = PortHandler(port_name)
        self.packet_handler = PacketHandler(2.0)
        self.group_sync_read = {}
        self.group_sync_write = {}
        for elem in ControlTable:
            self.group_sync_read[elem] = GroupSyncRead(
                self.port_handler,
                self.packet_handler,
                elem.value.address,
                elem.value.num_bytes,
            )
            self.group_sync_write[elem] = GroupSyncWrite(
                self.port_handler,
                self.packet_handler,
                elem.value.address,
                elem.value.num_bytes,
            )

        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {port_name}")

        if not self.port_handler.setBaudRate(baud_rate):
            raise RuntimeError(f"Failed to set baud rate {baud_rate}")

    def write(self, motor_id: int, control_table: ControlTable, data, retry_num=10):
        for _ in range(retry_num):
            try:
                match control_table.value.num_bytes:
                    case 1:
                        comm_result, error = self.packet_handler.write1ByteTxRx(
                            self.port_handler,
                            motor_id,
                            control_table.value.address,
                            data,
                        )
                    case 2:
                        comm_result, error = self.packet_handler.write2ByteTxRx(
                            self.port_handler,
                            motor_id,
                            control_table.value.address,
                            data,
                        )
                    case 4:
                        comm_result, error = self.packet_handler.write4ByteTxRx(
                            self.port_handler,
                            motor_id,
                            control_table.value.address,
                            data,
                        )
                    case _:
                        raise ValueError("Invalid num_bytes are detected")
            except IndexError:
                continue

            if comm_result == robotis_def.COMM_SUCCESS and error == 0:
                break

        if comm_result != robotis_def.COMM_SUCCESS:
            raise RuntimeError(
                f"Communication failed because of {self.packet_handler.getTxRxResult(comm_result)}"
            )
        if error != 0:
            raise RuntimeError(
                f"Write operation failed because of {self.packet_handler.getRxPacketError(error)}"
            )

    def read(self, motor_id: int, control_table: ControlTable, retry_num=10):
        for _ in range(retry_num):
            try:
                match control_table.value.num_bytes:
                    case 1:
                        value, comm_result, error = self.packet_handler.read1ByteTxRx(
                            self.port_handler, motor_id, control_table.value.address
                        )
                    case 2:
                        value, comm_result, error = self.packet_handler.read2ByteTxRx(
                            self.port_handler, motor_id, control_table.value.address
                        )
                    case 4:
                        value, comm_result, error = self.packet_handler.read4ByteTxRx(
                            self.port_handler, motor_id, control_table.value.address
                        )
                    case _:
                        raise ValueError("Invalid num_bytes are detected")
            except IndexError:
                continue

            if comm_result == robotis_def.COMM_SUCCESS and error == 0:
                break

        if comm_result != robotis_def.COMM_SUCCESS:
            raise RuntimeError(
                f"Communication failed because of {self.packet_handler.getTxRxResult(comm_result)}"
            )
        if error != 0:
            raise RuntimeError(
                f"Read operation failed because of {self.packet_handler.getRxPacketError(error)}"
            )

        return value

    def sync_read(
        self, motor_ids: List[int], control_table: ControlTable, retry_num=10
    ):
        self.group_sync_read[control_table].clearParam()
        for motor_id in motor_ids:
            self.group_sync_read[control_table].addParam(motor_id)

        for _ in range(retry_num):
            try:
                comm_result = self.group_sync_read[control_table].txRxPacket()
            except IndexError:
                continue

            if comm_result == robotis_def.COMM_SUCCESS:
                break

        if comm_result != robotis_def.COMM_SUCCESS:
            raise RuntimeError(
                f"Communication failed because of {self.packet_handler.getTxRxResult(comm_result)}"
            )

        return [
            self.group_sync_read[control_table].getData(
                motor_id, control_table.value.address, control_table.value.num_bytes
            )
            for motor_id in motor_ids
        ]

    def sync_write(
        self,
        motor_ids: List[int],
        control_table: ControlTable,
        data_list: List,
        retry_num=10,
    ):
        self.group_sync_write[control_table].clearParam()
        for motor_id, data in zip(motor_ids, data_list):
            match control_table.value.num_bytes:
                case 1:
                    binary_data = [data]
                case 2:
                    binary_data = [DXL_LOBYTE(data), DXL_HIBYTE(data)]
                case 4:
                    binary_data = [
                        DXL_LOBYTE(DXL_LOWORD(data)),
                        DXL_HIBYTE(DXL_LOWORD(data)),
                        DXL_LOBYTE(DXL_HIWORD(data)),
                        DXL_HIBYTE(DXL_HIWORD(data)),
                    ]
                case _:
                    raise ValueError(
                        f"Invalid num_bytes {control_table.value.num_bytes} are detected"
                    )

            self.group_sync_write[control_table].addParam(motor_id, binary_data)

        for _ in range(retry_num):
            try:
                comm_result = self.group_sync_write[control_table].txPacket()
            except IndexError:
                continue

            if comm_result == robotis_def.COMM_SUCCESS:
                break

        if comm_result != robotis_def.COMM_SUCCESS:
            raise RuntimeError(
                f"Communication failed because of {self.packet_handler.getTxRxResult(comm_result)}"
            )

    def reboot(self, motor_id: int):
        comm_result, error = self.packet_handler.reboot(self.port_handler, motor_id)
        if comm_result != robotis_def.COMM_SUCCESS:
            raise RuntimeError(
                f"Communication failed because of {self.packet_handler.getTxRxResult(comm_result)}"
            )
        if error != 0:
            raise RuntimeError(
                f"Reboot operation failed because of {self.packet_handler.getRxPacketError(error)}"
            )

    def close(self):
        self.port_handler.closePort()
