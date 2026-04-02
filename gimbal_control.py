import math
import time
import random
import sys

# Check if running in a virtual environment
if sys.prefix != sys.base_prefix:
    print("Warning: You are running in a virtual environment.")
    print("pigpio is a system package and may not be available in venv.")
    print("Try running with system Python: deactivate && python3 gimbal_control.py ...")
    print("Or install pigpio system-wide: sudo apt install python3-pigpio")
    # Continue anyway, in case it works

# Servo control mode
# By default we assume real hardware (pigpio + BNO055) is available.
# Use --mock to run without hardware.

MOCK_MODE = False
pi = None
pigpio = None
HAVE_PIGPIO = False

# Placeholder function to keep static analysis happy. Actual implementation
# is set when `_setup_servo_control()` is called.
def set_servo_position(pin, position):
    raise RuntimeError("Servo control not configured; call _setup_servo_control() first")

try:
    import pigpio  # or RPi.GPIO
    HAVE_PIGPIO = True
except ImportError:
    pigpio = None

# BNO055 library (optional)
bno55 = None
HAVE_BNO055 = False
BNO055_LIB = None
BNO055_SENSOR = None
BNO055_IMPORT_ERRORS = None

# Only support CircuitPython adafruit_bno055 for this build.
try:
    import board
    import adafruit_bno055 as bno55
    HAVE_BNO055 = True
    BNO055_LIB = "adafruit_bno055"
except ImportError as e:
    bno55 = None
    BNO055_IMPORT_ERRORS = (e,)


def _setup_servo_control(mock=None, require_hardware=False, servo_xz_pin=12, servo_yz_pin=13):
    """Configure servo output behavior.

    If mock is True, use stub functions and avoid pigpio.
    If mock is False, require pigpio and a running pigpio daemon.

    Args:
        mock: If True, force mock mode.
        require_hardware: If True, raise an error when pigpio daemon isn't available.
        servo_xz_pin: BCM pin number for the XZ servo (physical pin 32).
        servo_yz_pin: BCM pin number for the YZ servo (physical pin 33).
    """
    global MOCK_MODE, pi, SERVO_XZ, SERVO_YZ

    if mock is not None:
        MOCK_MODE = bool(mock)

    if MOCK_MODE:
        pi = None

        def set_servo_position(pin, position):
            print(f"Mock: Setting servo on pin {pin} to position {position}")

        # Expose mock function on module level for use by stabilize_gimbal.
        globals()["set_servo_position"] = set_servo_position

        SERVO_XZ = "XZ_SERVO"
        SERVO_YZ = "YZ_SERVO"
        return

    if not HAVE_PIGPIO:
        raise RuntimeError(
            "pigpio is not installed; install it or run with --mock for offline testing"
        )

    pi = pigpio.pi()
    if not getattr(pi, "connected", True):
        if require_hardware:
            raise RuntimeError(
                "pigpio daemon not running or failed to connect. "
                "Start it by running: `sudo pigpiod` or `sudo systemctl start pigpiod`"
            )

        # Fall back to mock mode if the pigpio daemon is not available.
        print(
            "Warning: pigpio daemon not running or failed to connect. "
            "Falling back to mock mode. Start it by running: `sudo pigpiod` or `sudo systemctl start pigpiod`"
        )
        MOCK_MODE = True
        pi = None

        def set_servo_position(pin, position):
            print(f"Mock: Setting servo on pin {pin} to position {position}")

        globals()["set_servo_position"] = set_servo_position
        SERVO_XZ = "XZ_SERVO"
        SERVO_YZ = "YZ_SERVO"
        return

    # Define servo pins
    SERVO_XZ = servo_xz_pin  # BCM pin for XZ plane servo (controls roll and yaw)
    SERVO_YZ = servo_yz_pin  # BCM pin for YZ plane servo (controls pitch and yaw)

    # Ensure pins are configured as outputs for servo PWM
    try:
        pi.set_mode(SERVO_XZ, pigpio.OUTPUT)
        pi.set_mode(SERVO_YZ, pigpio.OUTPUT)
    except Exception:
        # Some pigpio builds may not require explicit mode setting.
        pass


# Note: servo control is configured at runtime (e.g., via CLI flags) using
# `_setup_servo_control()`. This avoids import-time failures on platforms
# without pigpio.


def normalize_angle(angle):
    """
    Normalize an angle to the range -180 to 180 degrees.
    """
    return (angle + 180) % 360 - 180


class PIDController:
    """A simple PID controller.

    This implementation supports anti-windup via integrator clamping and
    optional output limits.
    """

    def __init__(self, kp, ki=0.0, kd=0.0, output_limits=(None, None), integrator_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._min_output, self._max_output = output_limits
        self._min_integrator, self._max_integrator = integrator_limits
        self.setpoint = 0.0

        self._integrator = 0.0
        self._prev_error = None

    def reset(self):
        self._integrator = 0.0
        self._prev_error = None

    def update(self, measurement, dt):
        """Compute PID output.

        Args:
            measurement: current value from sensor.
            dt: elapsed time in seconds since last update.

        Returns:
            Control output (same units as measurement).
        """
        if dt <= 0 or dt is None:
            return 0.0

        # error: setpoint - measurement
        error = normalize_angle(self.setpoint - measurement)

        # proportional term
        p = self.kp * error

        # integral term with anti-windup
        self._integrator += error * dt
        if self._min_integrator is not None:
            self._integrator = max(self._min_integrator, self._integrator)
        if self._max_integrator is not None:
            self._integrator = min(self._max_integrator, self._integrator)
        i = self.ki * self._integrator

        # derivative term
        d = 0.0
        if self._prev_error is not None:
            d = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        output = p + i + d
        if self._min_output is not None:
            output = max(self._min_output, output)
        if self._max_output is not None:
            output = min(self._max_output, output)

        return output


def random_unit_quaternion():
    """Generate a random unit quaternion (w, x, y, z)."""
    u1 = random.random()
    u2 = random.random()
    u3 = random.random()
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q0 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    return (q0, q1, q2, q3)


def quaternion_to_euler(quat, degrees=True):
    """Convert a quaternion (w, x, y, z) to Euler angles (heading/yaw, roll, pitch)."""
    w, x, y, z = quat

    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    if degrees:
        return math.degrees(yaw), math.degrees(roll), math.degrees(pitch)
    return yaw, roll, pitch


def euler_to_quaternion(euler, degrees=True):
    """Convert Euler angles (heading/yaw, roll, pitch) to a quaternion (w, x, y, z)."""
    heading, roll, pitch = euler
    if degrees:
        heading = math.radians(heading)
        roll = math.radians(roll)
        pitch = math.radians(pitch)

    cy = math.cos(heading * 0.5)
    sy = math.sin(heading * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp

    return (w, x, y, z)


def get_bno055_data():
    """Read the full sensor output from a BNO055 unit.

    Returns a tuple ``(quat, gyro, accel)`` where:

    * ``quat`` is the orientation quaternion ``(w, x, y, z)``.
    * ``gyro`` is angular rate around each axis in radians/sec.
    * ``accel`` is acceleration vector (gravity + linear motion) in m/s^2.

    If ``MOCK_MODE`` is True the function returns randomized data for
    offline testing.
    """
    if MOCK_MODE:
        quat = random_unit_quaternion()
        gyro = (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5))
        accel = (random.uniform(-16, 16), random.uniform(-16, 16), random.uniform(-16, 16))
        return quat, gyro, accel

    if not HAVE_BNO055:
        print("BNO055 library not installed; run with --mock or install a supported BNO055 package")
        if BNO055_IMPORT_ERRORS is not None:
            print("Import errors:")
            for idx, err in enumerate(BNO055_IMPORT_ERRORS, start=1):
                print(f"  {idx}. {type(err).__name__}: {err}")
        return None, None, None

    try:
        global BNO055_SENSOR

        if BNO055_LIB == "adafruit_bno055":
            if BNO055_SENSOR is None:
                try:
                    i2c = board.I2C()
                    BNO055_SENSOR = bno55.BNO055_I2C(i2c)
                except AttributeError:
                    # Some versions have the class named BNO055 and accept I2C
                    BNO055_SENSOR = bno55.BNO055(board.I2C())
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize adafruit_bno055 sensor: {e}")

            sensor = BNO055_SENSOR
            if sensor is None:
                raise RuntimeError("BNO055 sensor instance is not available")

            quat = getattr(sensor, "quaternion", None)
            if quat is None:
                euler = getattr(sensor, "euler", None)
                if euler is None:
                    raise RuntimeError("Failed to read orientation from adafruit_bno055")
                quat = euler_to_quaternion(euler)

            gyro = getattr(sensor, "gyro", None)
            accel = getattr(sensor, "acceleration", None)
            return quat, gyro, accel

        raise RuntimeError("Unsupported BNO055 library loaded; expected adafruit_bno055")
    except Exception as e:
        print(f"Error reading BNO055 sensor: {e}")
        return None, None, None

def get_gimbal_angles():
    """Convenience helper that returns normalized Euler angles for
    the gimbal control loop (computed from the quaternion).

    Returns:
        tuple: (x, y, z) where:
            x: roll angle (-180 to 180°)
            y: pitch angle (-180 to 180°)
            z: yaw angle (-180 to 180°)
    """
    quat, _, _ = get_bno055_data()
    if quat is None:
        return None, None, None

    heading, roll, pitch = quaternion_to_euler(quat)
    x = normalize_angle(roll)
    y = normalize_angle(pitch)
    z = normalize_angle(heading)
    return x, y, z

def angle_to_servo_position(angle, center_position=1500, max_deflection=333, servo_arm_length_mm=None):
    """
    Convert an angle in degrees to servo pulse width in microseconds.

    Args:
        angle: Desired gimbal angle in degrees (-60 to 60)
        center_position: Pulse width for center position (usually 1500us)
        max_deflection: Maximum deflection from center (333us for 60°)
        servo_arm_length_mm: Length of servo arm in mm (optional, for calibration notes)

    Returns:
        Pulse width in microseconds (typically 1167-1833)
    
    Note: The actual gimbal angle achieved depends on servo arm length and linkage geometry.
    For accurate control, calibrate by measuring actual gimbal deflection vs servo position.
    """
    # Map angle to servo position
    # Assuming servo range: -60° to 60° maps to center ± max_deflection
    # This is an approximation - actual gimbal angle = servo_angle * (arm_length / linkage_ratio)
    position = center_position + (angle / 60.0) * max_deflection

    # Clamp to valid servo range
    position = max(500, min(2500, position))

    return int(position)

def test_servo(pin, position_us, duration=2.0):
    """Test a servo by setting it to a fixed pulse width for a duration."""
    if MOCK_MODE:
        print(f"Mock: Setting servo on pin {pin} to {position_us}us for {duration}s")
        return True

    if pi is None:
        print("pigpio not configured. Call _setup_servo_control() first.")
        return False

    print(f"Setting servo on BCM pin {pin} to {position_us}us for {duration}s...")

    # Set pin as output (optional, pigpio handles it)
    pi.set_mode(pin, pigpio.OUTPUT)

    # Set servo pulse width
    ret = pi.set_servo_pulsewidth(pin, position_us)
    print(f"pigpio set_servo_pulsewidth( {pin}, {position_us} ) returned: {ret}")

    # Hold for duration
    time.sleep(duration)

    # Stop PWM (set to 0 to disable)
    ret2 = pi.set_servo_pulsewidth(pin, 0)
    print(f"pigpio stop servo returned: {ret2}")

    print("Test complete. Servo stopped.")
    return True


def stabilize_gimbal(
    roll_offset=0.0,
    pitch_offset=0.0,
    baseline_duration=5.0,
    baseline_rate=20,
):
    """Main gimbal stabilization loop using PID.

    This controller attempts to keep the gimbal "looking down" by keeping roll
    and pitch near zero (relative to gravity). The yaw (heading) is left free
    since the rings provide passive yaw stabilization.

    The hardware is arranged such that:
      * XZ servo (ring 1) adjusts roll (rotation around the X axis)
      * YZ servo (ring 2) adjusts pitch (rotation around the Y axis)

    The PID controllers compute a correction angle for each axis and map it
    to servo pulse widths.

    The first `baseline_duration` seconds are used to capture a baseline
    orientation (the pose to hold), which becomes the setpoint for stabilization.
    """
    _ensure_servo_configured()

    print("2-Axis Gimbal Stabilization Started (with rings for yaw)")
    print("XZ servo: controls roll, YZ servo: controls pitch")
    print("Rings handle yaw stabilization")
    print("Reading sensor data and adjusting servos... (Press Ctrl+C to stop)")

    # PID gains (tune these for your specific mechanics)
    roll_pid = PIDController(kp=1.5, ki=0.0, kd=0.1, output_limits=(-60, 60), integrator_limits=(-30, 30))
    pitch_pid = PIDController(kp=1.5, ki=0.0, kd=0.1, output_limits=(-60, 60), integrator_limits=(-30, 30))

    # Compute baseline orientation to hold (first few seconds)
    if baseline_duration > 0:
        print(f"Collecting baseline orientation for {baseline_duration:.1f} seconds...")
        baseline_samples = []
        interval = 1.0 / baseline_rate if baseline_rate > 0 else 0.05
        start_time = time.time()
        while time.time() - start_time < baseline_duration:
            roll, pitch, yaw = get_gimbal_angles()
            if roll is not None:
                baseline_samples.append((roll, pitch))
            time.sleep(interval)

        if baseline_samples:
            avg_roll = sum(r for r, _ in baseline_samples) / len(baseline_samples)
            avg_pitch = sum(p for _, p in baseline_samples) / len(baseline_samples)
        else:
            avg_roll = 0.0
            avg_pitch = 0.0

        print(
            f"Baseline orientation (avg over {len(baseline_samples)} samples): "
            f"roll={avg_roll:.2f}°, pitch={avg_pitch:.2f}°"
        )

        # Set the PID setpoints to the baseline orientation (+ offsets)
        roll_pid.setpoint = avg_roll + roll_offset
        pitch_pid.setpoint = avg_pitch + pitch_offset
    else:
        # Keep the gimbal "looking down" (roll=0, pitch=0)
        roll_pid.setpoint = roll_offset
        pitch_pid.setpoint = pitch_offset

    last_time = time.time()
    last_debug_time = last_time

    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

            # read full sensor output (quaternion, gyro and accel)
            quat, gyro, accel = get_bno055_data()
            if quat is not None:
                heading, roll, pitch = quaternion_to_euler(quat)

                # PID controller outputs a target gimbal angle (degrees)
                xz_target_angle = roll_pid.update(roll, dt) + roll_offset
                yz_target_angle = pitch_pid.update(pitch, dt) + pitch_offset

                xz_position = angle_to_servo_position(xz_target_angle)
                yz_position = angle_to_servo_position(yz_target_angle)

                if not MOCK_MODE:
                    pi.set_servo_pulsewidth(SERVO_XZ, xz_position)
                    pi.set_servo_pulsewidth(SERVO_YZ, yz_position)
                else:
                    set_servo_position(SERVO_XZ, xz_position)
                    set_servo_position(SERVO_YZ, yz_position)

                # Periodic debug output (once per second)
                if now - last_debug_time >= 1.0:
                    last_debug_time = now
                    print(
                        f"roll={roll:.1f}, pitch={pitch:.1f}, yaw={heading:.1f} | "
                        f"target_roll={xz_target_angle:.1f} ({xz_position}us), "
                        f"target_pitch={yz_target_angle:.1f} ({yz_position}us)"
                    )
                    if gyro is not None:
                        print(f"gyro rad/s={gyro}")
                    if accel is not None:
                        print(f"accel m/s^2={accel}")
            else:
                print("Failed to read sensor data")

            time.sleep(0.02)  # Update at ~50Hz

    except KeyboardInterrupt:
        print("\nStopping gimbal stabilization")
        if not MOCK_MODE:
            # Stop servos
            pi.set_servo_pulsewidth(SERVO_XZ, 0)
            pi.set_servo_pulsewidth(SERVO_YZ, 0)
            pi.stop()

def calibrate_servo_gimbal():
    """
    Calibration guide for gimbal servo control.
    
    To accurately control gimbal angles, you need to determine the relationship
    between servo pulse width and actual gimbal deflection angle.
    
    Steps:
    1. Mount servo and gimbal mechanism
    2. Set servo to center position (1500us)
    3. Measure gimbal angle (should be 0°)
    4. Set servo to max deflection (2000us for +90°, 1000us for -90°)
    5. Measure actual gimbal angle achieved
    6. Calculate scaling factor: actual_angle / 90°
    7. Adjust max_deflection in angle_to_servo_position accordingly
    
    Example: If servo at 2000us gives 75° instead of 90°, 
             max_deflection should be 500 * (75/90) = 417us
    """
    print("Gimbal Servo Calibration Guide:")
    print("1. Set servo to 1500us (center) - measure gimbal angle")
    print("2. Set servo to 1833us (+60°) - measure gimbal angle") 
    print("3. Set servo to 1167us (-60°) - measure gimbal angle")
    print("4. Calculate scaling: actual_angle / 60°")
    print("5. Update max_deflection = 333 * scaling_factor")
    print("6. Test with small angles and verify gimbal response")

# Uncomment to run calibration guide
# calibrate_servo_gimbal()

def print_servo_pulse_for_current_orientation():
    """Prints the servo pulse values for the current gimbal orientation (roll/pitch).
    Useful for confirming that 'looking down' corresponds to servo mid-position (1500us)."""
    roll, pitch, yaw = get_gimbal_angles()
    if roll is None:
        print("Unable to read gimbal orientation.")
        return
    xz_position = angle_to_servo_position(roll)
    yz_position = angle_to_servo_position(pitch)
    print(f"Current roll: {roll:.2f}°, pitch: {pitch:.2f}°")
    print(f"Servo XZ pulse: {xz_position}us (should be ~1500us for mid)")
    print(f"Servo YZ pulse: {yz_position}us (should be ~1500us for mid)")


def print_system_status():
    """Print environment / dependency status useful for debugging."""
    import sys
    import platform

    print("--- System status ---")
    print(f"Python: {sys.executable} ({platform.python_version()})")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")

    print(f"pigpio installed: {HAVE_PIGPIO}")
    if HAVE_PIGPIO:
        print(f"pigpio library path: {pigpio.__file__}")

    if HAVE_PIGPIO:
        # Check whether pigpio daemon can be reached
        try:
            pi_test = pigpio.pi()
            connected = getattr(pi_test, "connected", False)
            pi_test.stop()
        except Exception:
            connected = False
        print(f"pigpio daemon reachable: {connected}")

    print(f"pigpio daemon required: {'yes' if not MOCK_MODE else 'no (mock mode)'}")

    # Report servo pin configuration if set
    servo_xz = globals().get("SERVO_XZ")
    servo_yz = globals().get("SERVO_YZ")
    if servo_xz is not None and servo_yz is not None:
        print(f"Configured servo pins (BCM): XZ={servo_xz}, YZ={servo_yz}")

    print(f"BNO055 library installed: {HAVE_BNO055}")
    if HAVE_BNO055:
        try:
            print(f"bno055 library path: {bno55.__file__}")
        except Exception:
            pass
    else:
        if BNO055_IMPORT_ERRORS is not None:
            print("BNO055 import errors:")
            for idx, err in enumerate(BNO055_IMPORT_ERRORS, start=1):
                print(f"  {idx}. {type(err).__name__}: {err}")

    print("---------------------")


def run_sensor_reader(sample_rate=10, duration=None):
    """Continuously read and print BNO055 sensor data.

    Args:
        sample_rate: How often to sample the sensor (Hz).
        duration: How long to run (seconds). If None or <= 0, runs until interrupted.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    interval = 1.0 / sample_rate
    end_time = time.time() + duration if duration and duration > 0 else None

    print(f"Reading BNO055 sensor data at {sample_rate}Hz. Press Ctrl+C to stop.")
    try:
        while True:
            quat, gyro, accel = get_bno055_data()
            if quat is not None:
                heading, roll, pitch = quaternion_to_euler(quat)
                print(f"Quaternion: w={quat[0]:.4f}, x={quat[1]:.4f}, y={quat[2]:.4f}, z={quat[3]:.4f}")
                print(f"Euler (deg): heading={heading:.2f}, roll={roll:.2f}, pitch={pitch:.2f}")
                if gyro is not None:
                    print(f"Gyro (rad/s): x={gyro[0]:.3f}, y={gyro[1]:.3f}, z={gyro[2]:.3f}")
                if accel is not None:
                    print(f"Accel (m/s^2): x={accel[0]:.3f}, y={accel[1]:.3f}, z={accel[2]:.3f}")
            else:
                print("Failed to read sensor data")

            if end_time is not None and time.time() >= end_time:
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    print("\nStopped reading sensor data")


def calibrate_look_down(duration=5.0, sample_rate=20):
    """Compute roll/pitch offsets when gimbal is physically set to "look down".

    This helper assumes the gimbal is manually positioned to the desired down-
    facing configuration and then averages the measured orientation over
    `duration` seconds.

    Returns:
        (roll_offset, pitch_offset): values (in degrees) to add to the PID
        command such that the controller considers the current orientation "down".
    """
    if duration <= 0 or sample_rate <= 0:
        raise ValueError("duration and sample_rate must be positive")

    samples = []
    interval = 1.0 / sample_rate
    end_time = time.time() + duration

    print(f"Calibrating look-down orientation for {duration:.1f}s @ {sample_rate}Hz...")
    while time.time() < end_time:
        roll, pitch, yaw = get_gimbal_angles()
        if roll is None:
            raise RuntimeError("Unable to read gimbal orientation during calibration")
        samples.append((roll, pitch))
        time.sleep(interval)

    avg_roll = sum(r for r, _ in samples) / len(samples)
    avg_pitch = sum(p for _, p in samples) / len(samples)

    roll_offset = -avg_roll
    pitch_offset = -avg_pitch

    print(
        f"Calibration complete: avg roll={avg_roll:.2f}°, avg pitch={avg_pitch:.2f}°\n"
        f"Offsets to apply: roll_offset={roll_offset:.2f}°, pitch_offset={pitch_offset:.2f}°"
    )

    return roll_offset, pitch_offset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gimbal stabilization tools")
    parser.add_argument("--calibrate-down", action="store_true", help="Compute roll/pitch offsets for looking down")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds (calibration or sensor read)")
    parser.add_argument("--rate", type=int, default=20, help="Sampling rate (Hz)")
    parser.add_argument("--run", action="store_true", help="Start stabilization loop")
    parser.add_argument("--print-servo-pulse", action="store_true", help="Print servo pulse for current orientation")
    parser.add_argument("--read-sensor", action="store_true", help="Print raw BNO055 sensor data in a loop")
    parser.add_argument("--mock", action="store_true", help="Run without hardware (mock outputs)")
    parser.add_argument(
        "--servo-xz-pin",
        type=int,
        default=18,
        help="BCM GPIO pin for XZ servo (default: 18, physical pin 12).",
    )
    parser.add_argument(
        "--servo-yz-pin",
        type=int,
        default=19,
        help="BCM GPIO pin for YZ servo (default: 19, physical pin 35).",
    )
    parser.add_argument(
        "--baseline-duration",
        type=float,
        default=5.0,
        help="Seconds to sample baseline orientation at startup (default: 5.0).",
    )
    parser.add_argument(
        "--baseline-rate",
        type=int,
        default=20,
        help="Sampling rate (Hz) during baseline capture (default: 20).",
    )
    parser.add_argument(
        "--test-servo",
        nargs=2,
        metavar=("PIN", "POSITION"),
        help="Test servo: PIN (BCM) and POSITION (us, e.g., 1500 for center)"
    )
    parser.add_argument(
        "--test-duration",
        type=float,
        default=2.0,
        help="Duration for servo test in seconds (default: 2.0)"
    )
    parser.add_argument("--status", action="store_true", help="Print environment + dependency status")
    args = parser.parse_args()

    # Set up servo output mode (client may override to mock)
    _setup_servo_control(mock=args.mock, servo_xz_pin=args.servo_xz_pin, servo_yz_pin=args.servo_yz_pin)

    if args.status:
        print_system_status()

    if args.calibrate_down:
        calibrate_look_down(duration=args.duration, sample_rate=args.rate)

    if args.read_sensor:
        run_sensor_reader(sample_rate=args.rate, duration=args.duration)

    if args.print_servo_pulse:
        print_servo_pulse_for_current_orientation()

    if args.test_servo:
        pin = int(args.test_servo[0])
        position = int(args.test_servo[1])
        test_servo(pin, position, args.test_duration)

    if args.run:
        stabilize_gimbal(
            roll_offset=0.0,
            pitch_offset=0.0,
            baseline_duration=args.baseline_duration,
            baseline_rate=args.baseline_rate,
        )
