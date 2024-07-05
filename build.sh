#!/bin/bash

BUILD_DIR="build"

# Check if the build directory already exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir "$BUILD_DIR"
fi

# Navigate into the build directory
cd "$BUILD_DIR"

# Run CMake to configure the project
echo "Configuring project..."
cmake ..

# Build the project
echo "Building project..."
cmake --build .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build completed successfully."
else
    echo "Build failed."
    exit 1
fi

