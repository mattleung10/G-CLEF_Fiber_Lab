// May 2022
// AccelStepper for two motors
// WIth constant speed, switching directions, and nonblocking delay between direction changes
// Repinned May 18, 2022

#include <AccelStepper.h>

const int dirPin1 = 2;
const int stepPin1 = 3;
const int sleepreset1 = 4;
//const int reset1 = 5;
const int dirPin2 = 5;
const int stepPin2 = 6;
const int sleepreset2 = 7;
//const int reset2 = 13;
const int MS3 = 10;
const int MS2 = 9;
const int MS1 = 8;
const int _resolution[5][3] = {{0, 0, 0},{1, 0, 0},{0, 1, 0},{1, 1, 0},{1, 1, 1}};
const int micro_step = 2;
const int scale_factor = pow(2, micro_step);

int button_state = 0;
const int button_pin = A0;
const int relayPin = 11;

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
//AccelStepper stepper2 = AccelStepper(motorInterfaceType, stepPin2, dirPin2);
 
void setup() {
    Serial.begin(9600);
    Serial.println("\nHello world");
    Serial.println("arduino ready");
    
    pinMode(sleepreset1, OUTPUT);
    pinMode(sleepreset2, OUTPUT);
    digitalWrite(sleepreset1, HIGH);
    digitalWrite(sleepreset2, HIGH);

    //setup Microstep Select logic inputs
    pinMode(MS1, OUTPUT);
    pinMode(MS2, OUTPUT);
    pinMode(MS3, OUTPUT);
    digitalWrite(MS1, (int) _resolution[micro_step][0]);
    digitalWrite(MS2, (int) _resolution[micro_step][1]);
    digitalWrite(MS3, (int) _resolution[micro_step][2]);

    //Setup button
    pinMode(button_pin, INPUT);
    pinMode(relayPin, OUTPUT);
    digitalWrite(relayPin, LOW);
    while (true) { //wait until the button is pressed
        button_state = digitalRead(button_pin);
        if (button_state == HIGH) {
            digitalWrite(LED_BUILTIN, HIGH);
            break;
        }
    }
    digitalWrite(relayPin, HIGH);
    delay(500); //blocking delay, give some time for power supply voltage to go up

    //Set the maximum speed in steps/s
    stepper1.setMaxSpeed(4800); //max ~ 1200*4
    //stepper2.setMaxSpeed(4800);

    //Set the current position to 0
    stepper1.setCurrentPosition(0);
    //stepper2.setCurrentPosition(0);
    
    //Set the angular velocity
    omega_1 = (float)theta_1 / T_1;
    omega_2 = (float)theta_2 / T_2;

    Serial.println("theta_1 is:");
    Serial.println(theta_1);
    Serial.println("T_1 is:");
    Serial.println(T_1);
    Serial.println("omega_1 is:");
    Serial.println(omega_1);
    
    //stepper1.moveTo(theta_1);
    //stepper2.moveTo(theta_2);
    // setSpeed must be after moveTo, because moveTo recalculates the velocity values
    float nu = 1; //frequency [Hz]                                                      // <<<===== CHANGE THIS LINE AS REQUIRED!
    Serial.println(nu*200*scale_factor);
    stepper1.setSpeed(nu*200*scale_factor);
    //stepper2.setSpeed(omega_2);
    Serial.println(scale_factor);
}
 
void loop() {
    stepper1.runSpeed();
}
