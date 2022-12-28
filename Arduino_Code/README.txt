June 2022
Matthew Leung

The code in this directory is used for the mode scrambler prototype.
All the files are INO files (Arduino Sketches).

--------------------------------------------------------------------------------------------------

DIRECTORIES:

"Scrambling_Trial" contains code to control the stepper motors in the mode scrambler prototype.
This code is generalizable in that inside each Arduino Sketch is a class for controlling a motor.
This class can be instantiated many times, once for each motor, allowing for many motors to be
controlled simulataneously. The limitation is how many pins the Arduino model has.

"Test" contains various exploratory code.

"Test_Repin" contains a simple Arduino sketch, const_speed_test.ino, to make the stepper motor
rotate continuously in  one direction at a constant speed. This Arduino sketch was written after
the pins were changed in May 2022. const_speed_test.ino works with the current setup and can be
used instead of the code in "Scrambling Trial" for simple testing.
