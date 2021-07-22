# 3D WiFi Ray Tracer Simulator

This repository contains the source code for a ray tracer simulator of the emission of a WiFi antenna in an indoor setting.

This is the result of my Masters Thesis, and is mainly an academical work not intended as a production ready code.

There are two versions of the main simulation loop: one intented to run in the CPU launching a ray after another and the other one using a Nvidia Graphics card and its parallel properties, launching several rays at once.

## Building
Build with 

 ```cmake -DCMAKE_CUDA_FLAGS="arch_sm52" ```

 Be careful, the numbers after sm are the CUDA Compute Capabilities for the card to be used.