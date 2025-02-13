FROM ros:humble

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    trimesh \
    ultralytics \
    opencv-python \
    pyquaternion \
    pillow \
    rospkg \
    sensor_msgs \
    geometry_msgs \
    cv_bridge \
    rclpy

# Set the entry point
CMD ["python3", "main.py"]

