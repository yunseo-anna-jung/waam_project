"""
printer/port_scanner.py
Apple Silicon / macOS
"""

from serial.tools import list_ports


class PortScanner:

    @staticmethod
    def scan():

        ports = list_ports.comports()

        devices = []

        for port in ports:

            devices.append(
                {
                    "device": port.device,
                    "description": port.description,
                    "manufacturer": port.manufacturer,
                    "hwid": port.hwid,
                }
            )

        return devices