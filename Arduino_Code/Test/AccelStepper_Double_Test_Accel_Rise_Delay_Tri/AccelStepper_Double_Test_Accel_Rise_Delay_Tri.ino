// AccelStepper for two motors
// With constant rise and fall of angular velocity
// Here, I use different variables!
// With delay, nonblocking
// With triangular velocity profile

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

const float t_delay_1 = 25;
unsigned int theta_1 = 25 * scale_factor; //stepper 1 target position
unsigned int theta_2 = 200 * scale_factor; //stepper 2 target position
const unsigned int theta_1_og = theta_1;
const unsigned int theta_2_og = theta_2;
const float T_1 = 0.5 - t_delay_1/1000.0; //stepper 1 period [s] (how long it takes to complete the motion)
const float T_2 = 0.25; //stepper 2 period [s] (how long it takes to complete the motion)
const float t_r1 = T_1/2.0; //stepper 1 acceleration rise time [s] (same as the fall time)
const float t_r2 = T_2/2.0; //stepper 1 acceleration rise time [s] (same as the fall time)
int omega_1 = 0; //stepper 1 velocity [steps/s]
int omega_2 = 0; //stepper 2 velocity [steps/s]
double omegadot_1 = 0; //stepper 1 acceleration [steps/s^2]
double omegadot_2 = 0; //stepper 2 acceleration [steps/s^2]

boolean theta_1_reached = false;
long t_at_reached_1 = 0.0;
long te_1 = 0.0;


AccelStepper stepper1 = AccelStepper(motorInterfaceType, stepPin1, dirPin1);
AccelStepper stepper2 = AccelStepper(motorInterfaceType, stepPin2, dirPin2);
 
void setup() {
    Serial.begin(9600);
    Serial.println("\nHello world");
    Serial.println("arduino ready");
    
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

    //Set the maximum speed in steps/s
    stepper1.setMaxSpeed(4800); //max ~ 1200*4
    stepper2.setMaxSpeed(4800);

    //Set the current position to 0
    stepper1.setCurrentPosition(0);
    stepper2.setCurrentPosition(0);
    
    //Set the angular velocity and acceleration
    omega_1 = (int) ((float)theta_1 / (T_1 - t_r1));
    omega_2 = (int) ((float)theta_2 / (T_2 - t_r2));
    omegadot_1 = (double) ((float)theta_1 / (t_r1 * (T_1 - t_r1)));
    omegadot_2 = (double) ((float)theta_2 / (t_r2 * (T_2 - t_r2)));

    Serial.println("theta_1 is:");
    Serial.println(theta_1);
    Serial.println("t_r1 is:");
    Serial.println(t_r1);
    Serial.println("T_1 is:");
    Serial.println(T_1);
    Serial.println("T_1-t_r1 is:");
    Serial.println((float)theta_1 / (t_r1 * (T_1 - t_r1)));
    Serial.println("omega_1 is:");
    Serial.println(omega_1);
    Serial.println("omegadot_1 is:");
    Serial.println(omegadot_1);
    
    stepper1.moveTo(theta_1);
    stepper2.moveTo(theta_2);
    // setSpeed and setAcceleration must be after moveTo, because moveTo recalculates the velocity values
    stepper1.setSpeed(omega_1);
    stepper2.setSpeed(omega_2);
    stepper1.setAcceleration(omegadot_1);
    stepper2.setAcceleration(omegadot_2);
    Serial.print(scale_factor);
}
 
void loop() {
    if ((stepper1.distanceToGo() == 0) || (theta_1_reached == true)) {
        if (theta_1_reached == false) {
            theta_1_reached = true;
            t_at_reached_1 = millis();
        } else {
            if ((millis() - t_at_reached_1) > t_delay_1) {
                omega_1 = -1 * omega_1;
                omegadot_1 = -1 * omegadot_1;
                if (theta_1 == theta_1_og) {theta_1 = 0;}
                else {theta_1 = theta_1_og;}
                //theta_1 = -1 * theta_1;
                stepper1.moveTo(theta_1);
                stepper1.setSpeed(omega_1);
                stepper1.setAcceleration(omegadot_1);
                theta_1_reached = false;
            }
        }
    }
    if (stepper2.distanceToGo() == 0) {
        omega_2 = -1 * omega_2;
        omegadot_2 = -1 * omegadot_2;
        if (theta_2 == theta_2_og) {theta_2 = 0;}
        else {theta_2 = theta_2_og;}
        //theta_2 = -1 * theta_2;
        stepper2.moveTo(theta_2);
        stepper2.setSpeed(omega_2);
        stepper2.setAcceleration(omegadot_2);
    }
    stepper1.run();
    stepper2.run();
}
