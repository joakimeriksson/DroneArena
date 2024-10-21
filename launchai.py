#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket, struct, time
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import cflib.crtp  # to scan for Crazyflies instances
from cflib.crazyflie import Crazyflie  # to easily connect/send/receive data from a Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie  # wrapper around the “normal” Crazyflie class.
from cflib.utils import uri_helper  # to help connecting to a Crazyflie with a URI
from threading import Event, Thread, Timer
import sys
import multiprocessing
#from Utility.log import Log  # to log position
import glob
import face


from cflib.positioning.motion_commander import MotionCommander  # to help moving the drone
from cflib.utils.multiranger import Multiranger  # to use the Multiranger deck

import logging
import time

try:
    TEAM = int(sys.argv[1])  # We cast to int to check it's a number
except:
    print('Using default Team 1. Launch with ./launch.py 2 for Team 2, etc.')
    TEAM = 1

# URI to the Crazyflie to connect to.
uri = uri_helper.uri_from_env(default='radio://0/'+str(TEAM)+'0/2M/E7E7E7E7E7')
MIN_DISTANCE = 0.3  # m
DEFAULT_HEIGHT = 1
WAIT_UPDOWN = 1  # How long it go up or down (unit: 1/10 seconds)
WAIT_LEFTRIGHT = 1  # How long it goes left or right (unit: 1/10 seconds)
RADIUS = .1 # Radius when dancing in circle (unit: m)
IDLE = 3.0  # Idle time before moving (unit: s)
NUM_SEQ = 4  # How many sequences
SPEED = 0.5
SPEED_TAKING_OFF = .3  # Speed taking off
PUSHING_DOWN = 4   # How much pushed down before landing
DANCE = False

FLY = False
FACE = True

SAVE = False
DEBUG = True


# Wifi streaming
stream_w = 324
stream_h = 244
deck_port = 5000
deck_ip = "192.168.4.1"


deck_attached_event = Event()
dancing = Event()
# Prevents printing warnings from the logging framework (only errors)
logging.basicConfig(level=logging.ERROR)
start_time = time.time()

lock = multiprocessing.Lock()

def pprint(text):
    if DEBUG:
        print(text)

def rx_bytes(size, client_socket):
  data = bytearray()
  while len(data) < size:
    data.extend(client_socket.recv(size-len(data)))
  return data

def param_deck_flow(_, value_str):
    value = int(value_str)
    if value:
        deck_attached_event.set()

def is_close(range):
    if range is None:
        return False
    elif range < MIN_DISTANCE:
        # if we're dancing, abort!
        if dancing.is_set():
            pprint('Dancing sequence aborted')
            dancing.clear()
        start_time = time.time()
        return True
    else:
        return False

def wifi(lock, known_faces):
    # block for a moment
    # display a message
    print('Wifi process started.')
    lock.acquire()
    pprint("Connecting to socket on {}:{}...".format(deck_ip, deck_port))
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((deck_ip, deck_port))
        pprint("Socket connected")

        start = time.time()
        count = 0
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', stream_h, stream_w)
        bayer_img = np.zeros((stream_h, stream_w, 1), dtype=np.uint8)
        cv2.imshow('Video', bayer_img)
        lock.release()

        face_locations = []
        face_names = []
        process_this_frame = True

        while (1):
            # for i in range(10):
            # First get the info
            packetInfoRaw = rx_bytes(4, client_socket)
            [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
            # pprint("Length is {}".format(length))
            # pprint("Route is 0x{:02X}->0x{:02X}".format(routing & 0xF, routing >> 4))
            # pprint("Function is 0x{:02X}".format(function))
            # pprint('FUNCTION', function)

            # H unsigned short
            # h signed short
            # B unsigned char
            # b signed char

            header = rx_bytes(1, client_socket)
            [magic] = struct.unpack('<B', header)
            content = rx_bytes(length - 3, client_socket)  # packet length without CPX header (2 bytes) and magic value (1 byte)
            if magic == 0xBC:
                [width, height, depth, format, size] = struct.unpack('<HHBBI', content)
                # pprint("Magic is good")
                # pprint("Resolution is {}x{} with depth of {} byte(s)".format(width, height, depth))
                # pprint("Image format is {}".format(format))
                # pprint("Image size is {} bytes".format(size))

                # Now we start rx the image, this will be split up in packages of some size
                imgStream = bytearray()

                while len(imgStream) < size:
                    packetInfoRaw = rx_bytes(4, client_socket)
                    [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
                    # pprint("Chunk size is {} ({:02X}->{:02X})".format(length, src, dst))
                    chunk = rx_bytes(length - 2, client_socket)
                    imgStream.extend(chunk)

                count = count + 1
                meanTimePerImage = (time.time() - start) / count
                # pprint("{}".format(meanTimePerImage))
                # pprint("{} Hz".format(1/meanTimePerImage))

                bayer_img = np.frombuffer(imgStream, dtype=np.uint8)
                bayer_img.shape = (stream_h, stream_w)
                bayer_img = Image.fromarray(bayer_img)
                draw = ImageDraw.Draw(bayer_img)
                #font = ImageFont.truetype("DejaVuSans.ttf", 10)
                bayer_img = np.asarray(bayer_img)

                if SAVE:
                    cv2.imwrite(f"stream_out/raw/img_{count:06d}.png", bayer_img)

                frame = cv2.cvtColor(bayer_img, cv2.COLOR_GRAY2RGB)

                # Only process every other frame of video to save time
                if process_this_frame:
                    face.face_recognition(frame, known_faces)
                    # Display the resulting image
                    cv2.imshow('Video', frame)

                process_this_frame = not process_this_frame

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                continue

            if magic == 0xBD:
                background, hand = struct.unpack('<hh', content)
                continue

            pprint("Didn't understand... {}".format(magic))
    except KeyboardInterrupt:
        lock.release()
        pass

# Wait in 10th of second
def wait(t):
    for j in range(t):
        if dancing.is_set():
            time.sleep(.1)

def execute(sequence):
    dancing.set()
    pprint('Dancing Sequence {}'.format(sequence))

    if sequence == 0:
        mc.start_circle_left(RADIUS, SPEED)

    if sequence == 1:
        for i in range(3):
            if dancing.is_set():
                mc.start_up(.5)
            wait(WAIT_UPDOWN)
            if dancing.is_set():
                mc.start_down(.5)
            wait(WAIT_UPDOWN)
        if dancing.is_set():
            mc.stop()
        dancing.clear()

    if sequence == 2:
        mc.start_circle_right(RADIUS, SPEED)

    if sequence == 3:
        for i in range(3):
            if dancing.is_set():
                mc.start_left(SPEED)
            wait(WAIT_LEFTRIGHT)
            if dancing.is_set():
                mc.start_right(SPEED)
            wait(WAIT_LEFTRIGHT)
        if dancing.is_set():
            mc.stop()
        dancing.clear()


if __name__ == '__main__':
    if FACE:
        print('face_init')
        known_faces = face.load_known_faces("photos")
        print('face_init done')
        process = multiprocessing.Process(target=wifi, args=(lock, known_faces))
        process.start()
        time.sleep(5)

        print('Waiting camera link')
        lock.acquire()
        print('Camera link up.')

    if FLY:
        cflib.crtp.init_drivers()  # Initialize the low-level drivers
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            #with Log(scf) as log:  # to start logging position (DO NOT REMOVE)

            print('Checking the Flow deck is attached.')
            scf.cf.param.add_update_callback(group="deck", name="bcFlow2", cb=param_deck_flow)
            if not deck_attached_event.wait(timeout=3):
                print('No flow deck detected!')
                sys.exit(1)
            print('Flow deck attached.')

            print('Take off.')
            taking_off = True
            pushing_down = PUSHING_DOWN
            with MotionCommander(scf, default_height=.1) as mc:
                with Multiranger(scf) as multi_ranger:
                    keep_flying = True
                    mc.start_up(SPEED_TAKING_OFF)
                    sequence = 0
                    try:
                        while(keep_flying):

                            # Stay away from obstacles
                            x, y = 0, 0
                            if is_close(multi_ranger.front):
                                pprint('Front')
                                x -= SPEED
                            if is_close(multi_ranger.back):
                                pprint('Back')
                                x += SPEED
                            if is_close(multi_ranger.left):
                                pprint('Left')
                                y -= SPEED
                            if is_close(multi_ranger.right):
                                pprint('Right')
                                y += SPEED

                            # If something is up...
                            if is_close(multi_ranger.up):
                                pushing_down -= 1
                                pprint('PUSHING_DOWN'.format(pushing_down))
                                # Go down
                                if pushing_down > 0:
                                    mc.down(.15)
                                # Or if it's been a while, land
                                else:
                                    keep_flying = False
                            # Nothing up, reset counter
                            else:
                                pushing_down = PUSHING_DOWN

                            # Something under
                            if is_close(multi_ranger.down) and not taking_off:
                                # We reached the floor on purpose!
                                if pushing_down < PUSHING_DOWN:
                                    pprint('Down')
                                    keep_flying = False
                                # We're too close to the floor, take off
                                else:
                                    taking_off = True



                            if not (x, y) == (0, 0):
                                mc.start_linear_motion(x, y, 0)
                            if dancing.is_set():
                                start_time = time.time()

                            time.sleep(.05)

                            # Are we in a taking off sequence?
                            if taking_off and multi_ranger.down is not None:
                                    dancing.clear()
                                    if multi_ranger.down >= DEFAULT_HEIGHT:
                                        taking_off = False
                                        pprint('Reached {} metres.'.format(DEFAULT_HEIGHT))
                                        start_time = time.time()
                                        mc.stop()
                                    else:
                                        print('Keep climbing')
                                        mc.start_linear_motion(x, y, SPEED_TAKING_OFF)
                                        time.sleep(.2)

                            # If we're doing nothing, and it's time to dance... Dance!
                            if DANCE and not taking_off and (time.time() - start_time > IDLE):
                                x = Thread(target=execute, args=(sequence%NUM_SEQ,), daemon=True)
                                x.start()
                                sequence += 1


                    except KeyboardInterrupt:
                        keep_flying = False
                    keep_landing = True
                    print('Landing.')
                    while(keep_landing):
                        if multi_ranger.down < .1:
                            break
                        mc.start_down(.5)
                        time.sleep(0.1)

        print('Landed.')
    else:
        print('Pretending to fly.')
        while(1):
            time.sleep(1)
    process.terminate()
