#!/usr/bin/env python3
"""
Simple servo test script for Raspberry Pi using pigpio.

This script allows you to test servo motors by setting them to fixed pulse widths.
Useful for verifying wiring, power, and basic servo response before running the full gimbal control.

Usage:
    python3 servo_test.py --pin 12 --position 1500 --duration 2
    python3 servo_test.py --pin 13 --position 1800 --duration 3

Arguments:
    --pin: BCM GPIO pin number (e.g., 12 for physical pin 32)
    --position: Pulse width in microseconds (typically 1200-1800 for ±60° from center)
    --duration: How long to hold the position (seconds, default 2)
    --stop: Stop PWM on the pin (set to 0us)

Note: Servos require external power (4.8-6V). Pi GPIO provides signal only.
For safety, limit to ±60° from center (approx 1200-1800us).
"""

import argparse
import time
import sys

# Check if running in a virtual environment
if sys.prefix != sys.base_prefix:
    print("Warning: You are running in a virtual environment.")
    print("pigpio is a system package and may not be available in venv.")
    print("Try running with system Python: deactivate && python3 servo_test.py ...")
    print("Or install pigpio system-wide: sudo apt install python3-pigpio")
    # Continue anyway, in case it works

try:
    import pigpio
except ImportError as e:
    print(f"Failed to import pigpio: {e}")
    print("Install with: sudo apt install python3-pigpio")
    sys.exit(1)

def test_servo(pin, position_us, duration=2.0):
    """Test a servo by setting it to a fixed pulse width for a duration."""
    print(f"Connecting to pigpio daemon...")
    pi = pigpio.pi()

    if not pi.connected:
        print("Failed to connect to pigpio daemon. Is it running? (sudo systemctl start pigpiod)")
        return False

    print(f"Setting servo on BCM pin {pin} to {position_us}us for {duration}s...")

    # Set pin as output (optional, pigpio handles it)
    pi.set_mode(pin, pigpio.OUTPUT)

    # Set servo pulse width
    pi.set_servo_pulsewidth(pin, position_us)

    # Hold for duration
    time.sleep(duration)

    # Stop PWM (set to 0 to disable)
    pi.set_servo_pulsewidth(pin, 0)

    pi.stop()
    print("Test complete. Servo stopped.")
    return True

def stop_servo(pin):
    """Stop PWM on a servo pin."""
    print(f"Stopping servo on BCM pin {pin}...")
    pi = pigpio.pi()

    if not pi.connected:
        print("Failed to connect to pigpio daemon.")
        return False

    pi.set_servo_pulsewidth(pin, 0)
    pi.stop()
    print("Servo stopped.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test servo motors with pigpio")
    parser.add_argument(
        "--pin",
        type=int,
        required=True,
        help="BCM GPIO pin number for the servo (e.g., 12 for physical pin 32)"
    )
    parser.add_argument(
        "--position",
        type=int,
        help="Pulse width in microseconds (500-2500). Omit to stop servo."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration to hold position in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop PWM on the pin (equivalent to --position 0)"
    )

    args = parser.parse_args()

    if args.stop or args.position is None:
        success = stop_servo(args.pin)
    else:
        success = test_servo(args.pin, args.position, args.duration)

    sys.exit(0 if success else 1)