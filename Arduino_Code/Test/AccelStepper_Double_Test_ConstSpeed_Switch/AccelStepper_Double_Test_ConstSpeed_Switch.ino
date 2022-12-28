// AccelStepper for two motors
// Motors move at cosntant speed, and switch directions

#include <AccelStepper.h>

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

int stepper1speed = 200 * scale_factor;
int stepper2speed = 200 * scale_factor;
int stepper1targetpos = 100 * scale_factor;
int stepper2targetpos = 150 * scale_factor;

AccelStepper stepper1 = AccelStepper(motorInterfaceType, stepPin1, dirPin1);
AccelStepper stepper2 = AccelStepper(motorInterfaceType, stepPin2, dirPin2);
 
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
    stepper1.setMaxSpeed(4800); //max ~ 1200*4
    stepper2.setMaxSpeed(4800);

    stepper1.setCurrentPosition(0);
    stepper2.setCurrentPosition(0);

    stepper1.moveTo(100);
    stepper1.setSpeed(stepper1speed);
    stepper2.setSpeed(stepper2speed);
    Serial.print(scale_factor);
}
 
void loop() {
    //Serial.println(stepper1.currentPosition());

//if (stepper1.currentPosition() == abs(stepper1targetpos/2)) {
    
    if (stepper1.distanceToGo() == 0) {
        stepper1speed = -1 * stepper1speed;
        stepper1targetpos = -1 * stepper1targetpos;
        stepper1.moveTo(stepper1targetpos);
        stepper1.setSpeed(stepper1speed);
    } /*else if (stepper1.currentPosition() == -abs(stepper1targetpos/2)) {
        stepper1speed = -1 * stepper1speed;
        stepper1targetpos = -1 * stepper1targetpos;
        stepper1.moveTo(stepper1targetpos);
        stepper1.setSpeed(stepper1speed);
    }*/
    //stepper1.runSpeedToPosition();
    stepper1.runSpeed();
    stepper2.setSpeed(stepper2speed);
    stepper2.runSpeed();
}
