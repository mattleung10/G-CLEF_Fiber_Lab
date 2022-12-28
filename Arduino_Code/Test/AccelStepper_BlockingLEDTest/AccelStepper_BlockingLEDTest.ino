// https://forum.arduino.cc/t/simultaneous-stepper-motor-control-using-accelstepper/625179/2
//  inspired by https://www.makerguides.com/28byj-48-stepper-motor-arduino-tutorial/
//  19 august 2019

// the accelstepper library... non-blocking
// trying 2 motors

/*
   Example sketch to control a 28BYJ-48 stepper motor with ULN2003 driver board, AccelStepper and Arduino UNO.
  More info: https://www.makerguides.com
*/

// Include the AccelStepper library:
#include <AccelStepper.h>

//bwod to see if it blocks or not
unsigned long previousBlink;
int blinkTerval = 250;
bool blinkState = false;
int blinkPin = LED_BUILTIN;

// going to use a button to change speed on the fly
int basicSpeed = 500;
int motor1speed;
int motor2speed;
byte speedPin = 2;

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
const int micro_step = 1;
const int scale_factor = pow(2, micro_step);

const int motorInterfaceType = AccelStepper::DRIVER; //this has a value of 1; means a stepper driver (with Step and Direction pins)


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

    Serial.println(".... Accelstepper constant speed, 2 motors  ....");
    Serial.print("Compiler: ");
    Serial.print(__VERSION__);
    Serial.print(", Arduino IDE: ");
    Serial.println(ARDUINO);
    Serial.print("Created: ");
    Serial.print(__TIME__);
    Serial.print(", ");
    Serial.println(__DATE__);
    Serial.println(__FILE__);

    //turn off L13
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);
    pinMode(blinkPin, OUTPUT);
    digitalWrite(blinkPin, blinkState);

    // Set the maximum steps per second:
    stepper1.setMaxSpeed(1000);
    stepper2.setMaxSpeed(1000);

    Serial.println("\nsetup() done\n");

} //setup

void loop() {
    getSpeed(); //depends on button
    doSteps();
    doPulse();
} //loop

void doPulse()
{
  if (millis() - previousBlink >= blinkTerval)
  {
    previousBlink = millis();
    blinkState = !blinkState;
    digitalWrite(blinkPin, blinkState);
  }
}//pulse

void getSpeed() {
    motor1speed = 1.0*basicSpeed;
    motor2speed = 1.0*basicSpeed;
    blinkTerval=100;
    stepper2.setSpeed(motor1speed);
    stepper1.setSpeed(motor2speed);
}//getspeed

void doSteps()
{
  // Step the motor with constant speed as set by setSpeed():
  stepper2.runSpeed();
  stepper1.runSpeed();
}//steps
