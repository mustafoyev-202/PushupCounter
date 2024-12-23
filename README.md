# AI Push-Up Counter 💪🤖

## Overview

This is an AI-powered push-up counting application that uses computer vision and pose detection to automatically track and count push-ups from video input.

## Features

- Real-time push-up detection
- Accurate repetition counting
- Visualization of body angles
- Video input and output support

## Prerequisites

- Python 3.7+
- OpenCV
- NumPy
- cvzone
- PoseDetector module

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mustafoyev-202/PushupCounter.git
cd pushup-counter
```

2. Install required dependencies:
```bash
pip install opencv-python numpy cvzone
```

## Usage

```bash
python pushup_demo.py input_video.mp4 -o output_video.mp4
```

### Arguments
- `input_video.mp4`: Path to your input video
- `-o output_video.mp4`: (Optional) Output video path (default: pushup_counter_output.mp4)

## Demo Video

[View Demo Video](https://github.com/mustafoyev-202/PushupCounter/blob/main/output_video.mp4)

## How It Works

The script uses OpenCV and cvzone's PoseDetector to:
- Detect body landmarks
- Calculate arm and body angles
- Count push-up repetitions
- Annotate the video with push-up count and angle visualizations

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.