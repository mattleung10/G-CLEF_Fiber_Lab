// AccelStepper for two motors//
// May 2022
// Using MultiStepper library. Note: MultiStepper library only allows constant speed
/*
 * From the documentation: https://www.airspayce.com/mikem/arduino/AccelStepper/classMultiStepper.html
 * MultiStepper class can manage multiple AccelSteppers (up to MULTISTEPPER_MAX_STEPPERS = 10),
 * and cause them all to move to selected positions at such a (constant) speed that they
 * all arrive at their target position at the same time.
 * 
 * Motors arrive at target position at the same time!
 */

#include <AccelStepper.h>
#include <MultiStepper.h>

const int dirPin1 = 2;
const int stepPin1 = 3;
const int sleep1 = 4;
const int reset1 = 5;
const int dirPin2 = 7;
const int stepPin2 = 11;
const int sleep2 = 12;
const int reset2 = 13;
const int MS3 = 10;
const int MS2 = 9;
const int MS1 = 8;
const int _resolution[5][3] = {{0, 0, 0},{1, 0, 0},{0, 1, 0},{1, 1, 0},{1, 1, 1}};
const int micro_step = 0;
const int scale_factor = pow(2, micro_step);

const int motorInterfaceType = AccelStepper::DRIVER; //this has a value of 1; means a stepper driver (with Step and Direction pins)


AccelStepper stepper1 = AccelStepper(motorInterfaceType, stepPin1, dirPin1);
AccelStepper stepper2 = AccelStepper(motorInterfaceType, stepPin2, dirPin2);
MultiStepper steppers;
 
void setup() {
    Serial.begin(9600);
  
    pinMode(sleep1, OUTPUT);
    pinMode(reset1, OUTPUT);
    pinMode(sleep2, OUTPUT);
    pinMode(reset2, OUTPUT);
    digitalWrite(sleep1, HIGH);
    digitalWrite(reset1, HIGH);
    digitalWrite(sleep2, HIGH);
    digitalWrite(reset2, HIGH);

    //setup Microstep Select logic inputs
    pinMode(MS1, OUTPUT);
    pinMode(MS2, OUTPUT);
    pinMode(MS3, OUTPUT);
    digitalWrite(MS1, (int) _resolution[micro_step][0]);
    digitalWrite(MS2, (int) _resolution[micro_step][1]);
    digitalWrite(MS3, (int) _resolution[micro_step][2]);

    // Set the maximum speed in steps per second:
    stepper1.setMaxSpeed(1000);
    stepper2.setMaxSpeed(1000);

    steppers.addStepper(stepper1);
    steppers.addStepper(stepper2);

    Serial.print(scale_factor);
}
 
void loop() {
  long positions[2]; // Array of desired stepper positions
  
  positions[0] = 1000;
  positions[1] = 50;
  steppers.moveTo(positions);
  steppers.runSpeedToPosition(); // Blocks until all are in position
  delay(1000);
  
  // Move to a different coordinate
  positions[0] = -100;
  positions[1] = 100;
  steppers.moveTo(positions);
  steppers.runSpeedToPosition(); // Blocks until all are in position
  delay(1000);
}
