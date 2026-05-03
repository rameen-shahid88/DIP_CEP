# Vision Pipeline for Road Scene Understanding 🚗👁️

A real-time computer vision pipeline built with **OpenCV**, **NumPy**, and **YOLOv8** that performs:

- Lane Detection  
- Object Detection  
- Obstacle Avoidance Suggestion  
- HUD (Heads-Up Display) Visualization  
- Batch Video Processing  

Designed for road-driving footage, dashcam videos, or self-driving system simulations.

---

# Features

## Lane Detection

Detects left and right road lane boundaries using:

- Grayscale conversion  
- Histogram equalization  
- Gaussian blur  
- Canny edge detection  
- Region of interest masking  
- Hough Line Transform  
- Temporal smoothing  

Outputs steering guidance:

- `LEFT`
- `RIGHT`
- `STRAIGHT`

---

## Object Detection (YOLOv8)

Uses **Ultralytics YOLOv8n** model for real-time obstacle recognition.

Detects:

- Cars  
- Trucks  
- Buses  
- People  
- Motorcycles  
- Bicycles  
- Traffic Lights  
- Stop Signs  
- Animals  
- Furniture / Obstacles  

Only important road-relevant classes are processed.

---

## Smart Avoidance Direction

System analyzes detected objects and suggests:

- `GO`
- `LEFT`
- `RIGHT`

Based on:

- Object size  
- Position in frame  
- Threat level  
- Road center occupancy  

---

## Heads-Up Display (HUD)

Real-time overlay showing:

| Section | Info |
|--------|------|
| Lane | Current lane steering direction |
| Turn(Object) | Avoidance recommendation |
| Objects | Number of detected objects |

---

## Responsive Video Scaling

Automatically resizes all videos to fit display window while preserving aspect ratio.

Supports:

- Landscape videos  
- Portrait phone videos  
- High-resolution footage  

# Tech Stack

- Python 3.x  
- OpenCV  
- NumPy  
- Ultralytics YOLOv8  
- Collections (deque)  
