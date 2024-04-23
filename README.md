1.Discussion on the Problem Statement
=====================================

In the context of autonomous vehicles, it is important to be able to observe the environment and detect objects so that the vehicle can act accordingly. These objects may include street signs, other vehicles, pedestrians, traffic lights, etc.

We previously created a program which takes a frame of a video and applies a variety of filters such as to identify lane markers in the frame. However, this program is very slow, which would be dangerous to pedestrians and other drivers if the vehicle is unable to react to its surroundings in time.

The main goal of this project is to accelerate the program such that we are able to load and process each frame before the next one is loaded to remove any delay between them. This will be done by parallelizing the program using the NVIDIA Jetson Orin Nano and other techniques learned throughout our Parallel Programming course over the past semester.

2.Proposed Solution
===================

A python code for lane detection has already been created in another course, from which we will work off of as a baseline. This code can be parallelized using the CuPy library, which converts python code into a CUDA-compatible program; ideally this will split the image data into segments such that the program can be run on different threads of the GPU in parallel. After the execution of this parallelized code, the resulting image data will be sent back to the CPU to be displayed.

We plan to use the knowledge and experience gained from the work we've done so far in our

Mechatronics Engineering program including python coding, image processing, and parallelizing

using CUDA. This project requires us to use CuPy to wrap the python code onto our NVIDIA Jetson Orin Nano, which is a new topic for everyone in our group and will require additional research and testing in order to finish with a successful project.

3.Theory
========

In our original lane detection code we use a number of openCV functions to apply filters and adjustments to the frames as they are accessed from the video file. The main cv functions are as follows: 

            cv2.cvtColor - converts colour frame to grayscale

            cv2.GaussianBlur - applies a blur filter to the frame

            cv2.Canny - reduces the frame to only edges

            cv2.bitwise - applies a defined crop 

            cv2.HaughLinesP - draws lines between identified line segments

While we can divide each frame and run the functions on tiles of the frame, the last step will be to apply Haughlines on the final processed frame. To get the most out of our acceleration we will want to parallelize as much of the program as possible. Thankfully most of these functions use a combination of matrix multiplication, addition, or convolution. Therefore we have confidence that our proposed solution is attainable. 

4.Implementation of the Solution
================================

1.  Jetson Orin Nano was successfully configured.Following the instructions provided [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro).
2.  Jetson will prioritize Python 3.8 which is already installed, so we can force the jetson to use version 3.9. 
3.  Install pip.
4.  Check for CuPy Library. And Install. 
5.  Include CuPy code into existing python code to enable parallelizing of the program.
6.  Testing: Add timer to calculate runtime per frame. If runtime is under 20 ms, the program is done running before the next frame is loaded (given a framerate of 50 fps), meaning we have achieved our goal.

![](blob:https://euangoddard.github.io/e11870b1-a7c7-4a92-9be1-a77ae980ee7f)

1.  If we cannot time just the frame we can implement a timer to check that the entire program runs within the 26 seconds of the total time of the video. If it is anything more that the original videos length then we know we are not running the lane detection in real time. 

5.Problems Encountered
======================

1.  Initially had some trouble flashing the SD card
2.  Terminal stopped working during our attempts to set the default version of Python to 3.9

6.Final Reflection & Learning Plan
==================================

**Amelia:** I would like to learn more about how to use the terminal.

**Ian:** I believe that we may need to implement Scipy and Cupy to parallelize the entire program. This will require a lot more work to rewrite some of the opencv code using Scipy to apply all the matrix functions in the correct order. Alternatively I believe that we may need to write the function again in C++ using opencv, which could help us implement the cuda kernels a little easier. 

 **Alex:** I would like to utilize the potential of AI model training for object detection that the Jetson Nano has to offer. Taking advantage of features like the accelerator's high-performance GPU architecture, optimized libraries, and unified memory, I believe that we can train AI models to detect objects like speed signs, enhancing autonomous driving technology.

We were able to run our original python program on the Jetson Nano, but not any implementation with CuPy. This led to us not being able to time its total duration, which we planned to do by measuring the full execution time and taking the average based on the number of frames. We wanted to adjust our original code such that it runs a function inside the loop and measures the time that this function takes to execute, giving us the execution time for one frame. After seeing the results we could have learned whether or not our implementation of an accelerator actually made our program more efficient by improving its execution speed.

After resolving our issues with CuPy and the Jetson Nano, our future goal is to implement object detection with other parts of our video, such as the speed signs on the side of the road. This is a crucial part of autonomous driving as the cars will need to adapt to changing speed limits, and should be configured with an accelerator to minimize the risk of missing a sign due to the program executing other code. Leveraging the NVIDIA Jetson Orin Nano's high-performance GPU architecture, we can harness the power of deep learning frameworks like TensorFlow or PyTorch to train and deploy models for object detection. By utilizing CuPy's array manipulation capabilities alongside the Jetson Orin Nano's hardware acceleration, we can achieve real-time processing of video streams, allowing for quick and accurate identification of the speed signs in our video.

Overall we gained great insight on parallelizing capable programs using edge based accelerators like the NVIDIA Jetson Orin Nano, from its set up to running code on it. With more time to familiarize ourselves with the Jetson and CuPy, we could have been able to successfully perform our goal of parallelizing the image processing needed for lane detection. Even if we had the ability to run CuPy on the Jetson, we didn't have the chance to explore what we would need to change to the code for it to run with CuPy.

7. References
==============

<https://docs.cupy.dev/en/v5.4.0/reference/ndimage.html#opencv-mode>

<https://docs.cupy.dev/en/stable/user_guide/basic.html>

<https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html>

<https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html>

<https://www.pixelstech.net/article/1353768112-Gaussian-Blur-Algorithm>

<https://opencv.org/platforms/cuda/>

<https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro>
