# CRF-ActivityRecognition

This repository contains the implementation of an **activity recognition** system using Conditional Random Fields (CRF)
to extract high-level activities from robotic sensor data. The extracted activities can be utilized for event log
generation in the context of process mining.

## Introduction

Activity recognition is a critical component of systems that rely on understanding sensor data to infer high-level
behaviors. This project focuses on leveraging **Conditional Random Fields (CRFs)** to model sequential dependencies in
sensor data for accurate activity recognition.

The recognized activities are used to generate **event logs**, which are essential inputs for **process mining** to
analyze robotic systems.

## Installation

Create a virtual environment
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies
```
pip install -r requirements.txt
```
