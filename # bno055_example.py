# bno055_example.py
# Minimal BNO055 I2C reader using smbus2 (or smbus). Reads Euler angles, temperature, and calibration.
# Usage: python3 bno055_example.py

try:
except ImportError:

BNO055_ADDRESS_A = 0x28
BNO055_ID = 0xA0

REG_CHIP_ID = 0x00
REG_OPR_MODE = 0x3D
REG_PWR_MODE = 0x3E
REG_SYS_TRIGGER = 0x3F
REG_TEMP = 0x34
REG_EULER_H_LSB = 0x1A
REG_CALIB_STAT = 0x35

OPR_MODE_CONFIG = 0x00
OPR_MODE_NDOF = 0x0C
PWR_MODE_NORMAL = 0x00

class BNO055:
    def __init__(self, busnum=1, address=BNO055_ADDRESS_A):
        self.bus = SMBus(busnum)
        self.address = address

    def _write_byte(self, reg, val):
        self.bus.write_byte_data(self.address, reg, val)

    def _read_bytes(self, reg, length):
        return self.bus.read_i2c_block_data(self.address, reg, length)

    def _read_byte(self, reg):
        return self.bus.read_byte_data(self.address, reg)

    def begin(self, timeout=2.0):
        start = time.time()
        # wait for chip ID
        while True:
            try:
                chip_id = self._read_byte(REG_CHIP_ID)
            except OSError:
                chip_id = None
            if chip_id == BNO055_ID or (time.time() - start) > timeout:
                break
            time.sleep(0.05)
        if chip_id != BNO055_ID:
            raise RuntimeError("BNO055 not found (chip id {}).".format(chip_id))

        # Switch to config mode to configure
        self._write_byte(REG_OPR_MODE, OPR_MODE_CONFIG)
        time.sleep(0.03)

        # Normal power
        self._write_byte(REG_PWR_MODE, PWR_MODE_NORMAL)
        time.sleep(0.01)
        # Clear sys trigger
        self._write_byte(REG_SYS_TRIGGER, 0x00)
        time.sleep(0.01)

        # Switch to NDOF fusion mode
        self._write_byte(REG_OPR_MODE, OPR_MODE_NDOF)
        time.sleep(0.02)

    @staticmethod