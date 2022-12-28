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
const int MAXSPEED = 4800;

AccelStepper stepper1 = AccelStepper(motorInterfaceType, stepPin, dirPin);
 
void setup() {
    Serial.begin(9600);
    digitalWrite(sleep, HIGH);
    digitalWrite(reset, HIGH);

    //setup Microstep Select logic inputs
    digitalWrite(MS1, (int) _resolution[micro_step][0]);
    digitalWrite(MS2, (int) _resolution[micro_step][1]);
    digitalWrite(MS3, (int) _resolution[micro_step][2]);

    // Set the maximum speed in steps per second:
    int SETSPEED = 400*scale_factor;
    if (SETSPEED > MAXSPEED) {
        Serial.print("Set speed is greater than max speed!");
    }
    stepper1.setMaxSpeed(MAXSPEED);
    stepper1.setAcceleration(50*scale_factor); //set the acceleration to 30 steps/s^2
    stepper1.setSpeed(SETSPEED); //set the target speed to 800 steps/s
    stepper1.moveTo(10000*scale_factor); //move to a position of 20000 steps

    Serial.print(scale_factor);
}
 
void loop() {
    /* There are several ways to use the AccelStepper library:
     *  
     *  METHOD 1:
     *  If you want a constant speed, in the main loop, just do:
     *  stepper1.setSpeed(200);
     *  stepper1.runSpeed(); //Poll the motor and step it if a step is due, implementing a constant speed as set by the most recent call to setSpeed().
     * 
     *  METHOD 2:
     *  If you want to move to a position POS with a speed SETSPEED and acceleration ACCEL, then: 
     *  In the setup, do:
     *  stepper1.setAcceleration(ACCEL);
     *  stepper1.setSpeed(SETSPEED);
     *  stepper1.moveTo(POS);
     *  In your main loop, just do:
     *  stepper1.run(); //Poll the motor and step it if a step is due, implementing accelerations and decelerations to achieve the target position. Takes only ONE step!
     */
     
    //stepper1.setSpeed(200*scale_factor);
    //stepper1.runSpeed();
    stepper1.run(); //take ONE step
}
