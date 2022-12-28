June 2022

The code in this directory is used for the mode scrambler prototype.
Although the Arduino sketches say "Single", the classes inside the sketches can be used to
control multiple motors.

--------------------------------------------------------------------------------------------------

WHAT EACH SKETCH DOES:

AccelStepper_Single_ConstSpeed.ino rotates the stepper motor from 0 degrees to some amplitude,
switches direction, and then goes back to 0 degrees. This motion is then repeated.
No accelarations are implemented. A stepper motor is not really intended for this kind of motion
at high frequencies (the motor will likely slip when it attempts to switch direction suddenly),
and so this sketch should not be used. 
This sketch contains a class called StepperConstSpeedSwitch which can be instantiated multiple
times to control multiple motors.

AccelStepper_Single_ConstSpeed_NoSwitch.ino rotates the stepper motor continuously in one
direction at a fixed speed. No accelerations are implemented. This sketch is recommended.
This sketch contains a class calledStepperConstSpeed which can be instantiated multiple
times to control multiple motors.
