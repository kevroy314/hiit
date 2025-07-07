#!/usr/bin/env python3
"""
HIIT Analyzer - Main Application Entry Point

A Streamlit application for analyzing High-Intensity Interval Training (HIIT) data
from FIT files, including heart rate analysis, interval detection, and performance metrics.
"""

import streamlit as st
from hiit.ui import main

if __name__ == "__main__":
    main() 