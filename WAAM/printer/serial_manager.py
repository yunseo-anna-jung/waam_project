"""
printer/serial_manager.py
"""

import serial

from logger.logger import logger


class SerialManager:

    def __init__(self):

        self.serial = None

    def connect(self, port, baudrate, timeout):

        try:

            self.serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout
            )

            logger.info(f"Connected : {port}")

            print(f"Connected : {port}")

            return True

        except Exception as e:

            logger.error(str(e))

            print(e)

            return False

    def disconnect(self):

        if self.serial:

            self.serial.close()

            logger.info("Serial Closed")

    def is_connected(self):

        if self.serial is None:

            return False

        return self.serial.is_open