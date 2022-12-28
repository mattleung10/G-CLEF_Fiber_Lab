// accelStepper example

#include <AccelStepper.h>

const int dirPin = 2;
const int stepPin = 3;
const int sleep = 4;
const int reset = 5;
const int MS3 = 10;
const int MS2 = 9;
const int MS1 = 8;
const int _resolution[5][3] = {{0, 0, 0},{1, 0, 0},{0, 1, 0},{1, 1, 0},{1, 1, 1}};
const int micro_step = 2;
const int scale_factor = pow(2, micro_step);

const int motorInterfaceType = AccelStepper::DRIVER; //this has a value of 1; means a stepper driver (with Step and Direction pins)

int stepper1speed = 200*scale_factor;
int stepper1targetpos = 100*scale_factor;

AccelStepper stepper1 = AccelStepper(motorInterfaceType, stepPin, dirPin);
 
void setup() {
    Serial.begin(9600);
    pinMode(sleep, OUTPUT);
    pinMode(reset, OUTPUT);
    pinMode(stepPin, OUTPUT);
    pinMode(dirPin, OUTPUT);
    digitalWrite(sleep, HIGH);
    digitalWrite(reset, HIGH);

    //setup Microstep Select logic inputs
    pinMode(MS1, OUTPUT);
    pinMode(MS2, OUTPUT);
    pinMode(MS3, OUTPUT);
    digitalWrite(MS1, (int) _resolution[micro_step][0]);
    digitalWrite(MS2, (int) _resolution[micro_step][1]);
    digitalWrite(MS3, (int) _resolution[micro_step][2]);

    // Set the maximum speed in steps per second:
    stepper1.setMaxSpeed(4800); //max ~ 1200*4 ; about 4000 steps should be your max

    stepper1.setCurrentPosition(0);
    stepper1.moveTo(stepper1targetpos);
    stepper1.setSpeed(stepper1speed);

    Serial.print(scale_factor);
}
 
void loop() {     
    if (stepper1.distanceToGo() == 0) {
        delay(200); //May 3, 2022
        stepper1speed = -1 * stepper1speed;
        stepper1targetpos = -1 * stepper1targetpos;
        //stepper1.moveTo(stepper1targetpos);
        //stepper1.setSpeed(stepper1speed);
    } 
    stepper1.runSpeedToPosition();
}
