#!/usr/bin/env python3
"""
Step 3: Run real-time keyword spotting from microphone.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from inference.realtime import main

if __name__ == "__main__":
    main()
