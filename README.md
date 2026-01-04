# Electric Speedster Jordan High School
Science Olympiad Electric Vehicle (2025–2026)

This repository contains the control code and supporting utilities for a **Science Olympiad Electric Vehicle** designed for precision distance travel, repeatability, and rule-compliant autonomy.

The system emphasizes:
- deterministic motion control  
- accurate distance targeting  
- consistent braking behavior  
- modular tuning for different track conditions  

---

## Overview

The Electric Vehicle event requires a battery-powered vehicle to travel a target distance as accurately as possible and stop within a designated scoring zone. This project focuses on **software-driven precision**, combining sensor feedback, calibrated motion profiles, and iterative testing.

Core design goals:
- minimize run-to-run variance  
- decouple tuning parameters from logic  
- allow fast adjustments between competition runs  

---

## Features

- **Distance-based motion control**
  - encoder-driven or time-based travel (configurable)
- **Predictable stopping behavior**
  - controlled deceleration and braking logic
- **Parameterized configuration**
  - speed, ramping, cutoff distance, and brake timing are easily adjustable
- **Fail-safe handling**
  - conservative defaults to avoid overshoot
- **Lightweight architecture**
  - optimized for microcontroller constraints

---

## Hardware Assumptions

This codebase is written to be hardware-agnostic but assumes:
- a DC motor drivetrain  
- motor driver with PWM control  
- wheel encoders *or* calibrated timing constants  
- microcontroller compatible with Arduino-style toolchains  

(Exact pin mappings and constants should be updated for your specific build.)

---

## Tuning Workflow

1. **Calibrate distance**
   - determine encoder ticks or time per unit distance
2. **Set cruise speed**
   - high enough for consistency, low enough for braking margin
3. **Adjust cutoff point**
   - when the vehicle transitions from drive → brake
4. **Iterate**
   - log results and refine parameters between runs

Most improvements come from **reducing variance**, not increasing speed.

---

## Attribution

This project is adapted from control logic originally developed by **Dillon Mehta** and **Jay Sahni** for the 2024 Science Olympiad **Robot Tour** event, which placed **2nd overall** at a national-level invitational.

The code was subsequently **reworked, extended, and adapted by Dillon Mehta** for use in his **2025–2026 Science Olympiad Electric Vehicle**, which he currently competes with. Modifications reflect the Electric Vehicle event’s distinct constraints, scoring mechanics, and emphasis on braking precision and distance repeatability.

---

## Disclaimer

This repository is intended for **educational and competition use**.  
Users are responsible for ensuring compliance with the official Science Olympiad rules for their competition year.
