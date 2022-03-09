import asyncio
import logging
import logging.handlers as handlers
import os

from bleak import discover
from bleak import BleakClient

devices_dict = {}
devices_list = []
receive_data = []
filename = "log.txt"
log = logging.getLogger(__name__)

#To discover BLE devices nearby
async def scan():
    dev = await discover()
    for i in range(0,len(dev)):
        #Print the devices discovered
        print("[" + str(i) + "]" + dev[i].address,dev[i].name,dev[i].metadata["uuids"])
        #Put devices information into list
        devices_dict[dev[i].address] = []
        devices_dict[dev[i].address].append(dev[i].name)
        devices_dict[dev[i].address].append(dev[i].metadata["uuids"])
        devices_list.append(dev[i].address)

#An easy notify function, just print the receive data
def notification_handler(sender, data):
    log.info(', '.join('{:03d}'.format(x) for x in data))


async def run(address, debug=False):
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(message)s")

    # https://docs.python.org/ko/3/library/logging.handlers.html
    should_roll_over = os.path.isfile(filename)
    handler = logging.handlers.RotatingFileHandler(filename, mode='w', backupCount=5)
    if should_roll_over:  # log already exists, roll over!
        handler.doRollover()

    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    log.addHandler(handler)

    if debug:
        import sys

        log.setLevel(logging.DEBUG)
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.DEBUG)
        log.addHandler(h)

        async with BleakClient(address) as client:
            log.info("Connected: {0}".format(client.is_connected))

            #Characteristic uuid
            CHARACTERISTIC_UUID = "991F0001-7774-3332-15F5-90324778E1BF"

            await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
            await asyncio.sleep(60.0*1)
            await client.stop_notify(CHARACTERISTIC_UUID)

if __name__ == "__main__":
    print("Scanning for peripherals...")

    flag = True
    while(flag):
        try:
            #Build an event loop
            loop = asyncio.get_event_loop()
            #Run the discover event
            loop.run_until_complete(scan())

            #let user chose the device
            index = input('please select device from 0 to ' + str(len(devices_list)) + ":")
            index = int(index)
            address = devices_list[index]
            flag = False
        except Exception as e:
            devices_dict = {}
            devices_list = []
            receive_data = []
            pass

    print("Address is " + address)

    #Run notify event
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    loop.run_until_complete(run(address, True))