// Matthew Leung
// April 2022
// Single stepper motor test, with stepperClass.cpp

#include <stepperClass.h>

const int dirPin = 2;
const int stepPin = 3;
const int sleep = 4;
const int reset = 5;
const int MS3 = 10;
const int MS2 = 9;
const int MS1 = 8;

int delayMS = 100000; // delay between steps in microseconds
// int delayMS = 200000; // delay between steps in microseconds
stepperClass m1(dirPin, stepPin, sleep, reset, MS1, MS2, MS3, true);

void setup() {
    m1.setPIN();
    m1.SleepResetHigh();
    m1.setZero();
}

void loop() {
    m1.rot(45.0, 2, delayMS);
    delay(100);
    m1.rot(-45.0, 2, delayMS);
    delay(100);
}
