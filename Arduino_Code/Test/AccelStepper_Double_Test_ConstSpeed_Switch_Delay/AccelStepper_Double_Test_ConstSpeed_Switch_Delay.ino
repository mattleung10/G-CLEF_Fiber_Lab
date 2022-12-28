// AccelStepper for two motors
// WIth constant speed, switching directions, and nonblocking delay between direction changes

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
const int micro_step = 2;
const int scale_factor = pow(2, micro_step);

const int motorInterfaceType = AccelStepper::DRIVER; //this has a value of 1; means a stepper driver (with Step and Direction pins)

const float t_delay_1 = 100; //delay in [ms]
const float t_delay_2 = 100; //delay in [ms]
unsigned int theta_1 = 100 * scale_factor; //stepper 1 target position
unsigned int theta_2 = 200 * scale_factor; //stepper 2 target position
const unsigned int theta_1_og = theta_1;
const unsigned int theta_2_og = theta_2;
const float T_1 = 1.0 - t_delay_1/1000.0; //stepper 1 period [s]
const float T_2 = 1.0 - t_delay_2/1000.0; //stepper 2 period [s]
float omega_1 = 0; //stepper 1 velocity [steps/s]
float omega_2 = 0; //stepper 2 velocity [steps/s]

boolean theta_1_reached = false;
boolean theta_2_reached = false;
long t_at_reached_1 = 0.0;
long t_at_reached_2 = 0.0;
long te_1 = 0.0;
long te_2 = 0.0;

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
    
    //Set the angular velocity
    omega_1 = (float)theta_1 / T_1;
    omega_2 = (float)theta_2 / T_2;

    Serial.println("theta_1 is:");
    Serial.println(theta_1);
    Serial.println("T_1 is:");
    Serial.println(T_1);
    Serial.println("omega_1 is:");
    Serial.println(omega_1);
    
    stepper1.moveTo(theta_1);
    stepper2.moveTo(theta_2);
    // setSpeed must be after moveTo, because moveTo recalculates the velocity values
    stepper1.setSpeed(omega_1);
    stepper2.setSpeed(omega_2);
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
                if (theta_1 == theta_1_og) {theta_1 = 0;}
                else {theta_1 = theta_1_og;}
                stepper1.moveTo(theta_1);
                stepper1.setSpeed(omega_1);
                theta_1_reached = false;
            }
        }
    }

    if ((stepper2.distanceToGo() == 0) || (theta_2_reached == true)) {
        if (theta_2_reached == false) {
            theta_2_reached = true;
            t_at_reached_2 = millis();
        } else {
            if ((millis() - t_at_reached_2) > t_delay_2) {
                omega_2 = -1 * omega_2;
                if (theta_2 == theta_2_og) {theta_2 = 0;}
                else {theta_2 = theta_2_og;}
                stepper2.moveTo(theta_2);
                stepper2.setSpeed(omega_2);
                theta_2_reached = false;
            }
        }
    }
    
    /*
    if (stepper2.distanceToGo() == 0) {
        omega_2 = -1 * omega_2;
        if (theta_2 == theta_2_og) {theta_2 = 0;}
        else {theta_2 = theta_2_og;}
        //theta_2 = -1 * theta_2;
        stepper2.moveTo(theta_2);
        stepper2.setSpeed(omega_2);
    }
    */
    stepper1.runSpeedToPosition();
    stepper2.runSpeedToPosition();
}
