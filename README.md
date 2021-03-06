# Final Project for MECH296A
This is a group project for MECH296A at Santa Clara University in the Spring of 2022. Teammates:
- James Forman
- Rehan Nazir
- Krishna Chaitanya
## Task
A random goal is assigned at the start, which is is either on the blue side or green side as shown in the demo video.
The task is to move the robot to the goal, and then to the other side of the field to the opponent's goal. To achieve this run the below file in the terminal, type:
```
cd final_competition
python final_competition.py
```

It uses HSV color space to detect if the robot is looking at which color side goal. Finally MobileNetSSDv1 is fine tuned to detect the goal, ball and other robot.

To stop the robot from runing run
```
cd final_competition
python stop_robot.py
```

# Tips and Tricks
- While training the model for robot, goal or ball any of the other object should not be present in the frame. If present they should be annotated accordingly. This is one of the main issues due to which we had to annotate all the images manually.
- Use weights and biases to monitor the training and save the best epoch
- Compile OpenCV with (optional Qt) and CUDA support for maximum performance
- Make a virtual environment for the project. Don't litter the system interpreter with the project files.
- Connect the Jetson TO Eduroam and do X11 forwarding and use "export DISPLAY=:10" to get imshow windows directly to your PC. Use slack to get IP address on boot using crontab. Access Jetson using Remote-SSH extension in VSCode. Add the system to keys to Jetson for password less access. 

# Model Weights
In networks folder!
# Demo Video
We won the final :soccer: tournament with a score of 2-0 :zap: :tada: 
[![Final Robot Tournament](https://img.youtube.com/vi/KYHkfP-IXz8/0.jpg)](https://www.youtube.com/watch?v=KYHkfP-IXz8)
