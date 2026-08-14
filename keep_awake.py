# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 00:29:44 2026

@author: anjum
"""

import time
import ctypes

# Simple loop to prevent Windows from locking or sleeping
print("Keep-Awake script active. Press Ctrl+C to stop.")
while True:
    # Simulates an unassigned virtual key press (F15)
    ctypes.windll.user32.keybd_event(0x7E, 0, 0, 0)
    ctypes.windll.user32.keybd_event(0x7E, 0, 2, 0)
    time.sleep(60)
