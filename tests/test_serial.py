from config import BAUDRATE
from config import TIMEOUT

from printer.port_scanner import PortScanner
from printer.serial_manager import SerialManager


ports = PortScanner.scan()

if len(ports) == 0:

    print("No Serial Device")

    exit()

manager = SerialManager()

manager.connect(
    ports[0]["device"],
    BAUDRATE,
    TIMEOUT
)

print()

print("Connected :", manager.is_connected())