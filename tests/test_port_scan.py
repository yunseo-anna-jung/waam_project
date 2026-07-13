from printer.port_scanner import PortScanner

ports = PortScanner.scan()

print()

print("========== USB Devices ==========")

for p in ports:

    print("----------------------------")

    print("Device :", p["device"])

    print("Desc   :", p["description"])

    print("Maker  :", p["manufacturer"])

    print("HWID   :", p["hwid"])