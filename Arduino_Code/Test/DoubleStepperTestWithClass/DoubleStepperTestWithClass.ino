// Matthew Leung
// April 2022
// Double stepper motor test, with stepperClass.cpp
// The two motors DO NOT move at the same time because of stepperClass.cpp
// The two motors move one at a time

#include <stepperClass.h>

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

int delayMS = 400000; // delay between steps in microseconds
// int delayMS = 200000; // delay between steps in microseconds
stepperClass m1(dirPin1, stepPin1, sleep1, reset1, MS1, MS2, MS3, true);
stepperClass m2(dirPin2, stepPin2, sleep2, reset2, MS1, MS2, MS3, true);

void setup() {
    m1.setPIN();
    m1.SleepResetHigh();
    m1.setZero();
    m2.setPIN();
    m2.SleepResetHigh();
    m2.setZero();
}

void loop() {
    m1.rot(45.0, 2, delayMS);
    m2.rot(45.0, 2, delayMS);
    delay(100);
    m1.rot(-45.0, 2, delayMS);
    m2.rot(-45.0, 2, delayMS);
    delay(100);
}
