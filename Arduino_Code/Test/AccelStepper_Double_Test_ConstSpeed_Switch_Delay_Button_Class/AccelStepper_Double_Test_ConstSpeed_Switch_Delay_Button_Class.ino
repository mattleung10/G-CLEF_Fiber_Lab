// May 2022
// AccelStepper for two motors
// WIth constant speed, switching directions, and nonblocking delay between direction changes
// Repinned May 18, 2022

#include <AccelStepper.h>

class StepperConstSpeedSwitch {
    public:
        int dir_pin; //direction pin
        int step_pin; //step pin
        int sleep_reset_pin; //sleep and reset pin
        int MS1;
        int MS2;
        int MS3;
        
        bool are_pins_set;
        bool are_MS_pins_set;
        bool is_MS_set;
        
        int micro_step;
        int scale_factor;

        unsigned int theta;
        float omega;

        AccelStepper AccelStepperObject; 
        
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        StepperConstSpeedSwitch(int di, int st, int slre) :
        _resolution {{0, 0, 0},{1, 0, 0},{0, 1, 0},{1, 1, 0},{1, 1, 1}}
        {
            //Class constructor
            dir_pin = di;
            step_pin = st;
            sleep_reset_pin = slre;
            are_MS_pins_set = false;
            are_pins_set = false;
            is_MS_set = false;

            motorInterfaceType = AccelStepper::DRIVER; //this has a value of 1; means a stepper driver (with Step and Direction pins)
            //AccelStepperObject = AccelStepper(motorInterfaceType, step_pin, dir_pin);
            AccelStepperObject = AccelStepper(1, step_pin, dir_pin);
            AccelStepperObject.setMaxSpeed(4800); //Set the maximum speed in steps/s; max ~ 1200*4
            AccelStepperObject.setCurrentPosition(0); //Set the current position to 0

            _SPR = 200;
        }

        int set_pins() {
            pinMode(dir_pin, OUTPUT);
            pinMode(step_pin, OUTPUT);
            pinMode(sleep_reset_pin, OUTPUT);
            digitalWrite(sleep_reset_pin, HIGH);
            are_pins_set = true;
            return 0;
        }

        int set_microstep_pins(int ms1, int ms2, int ms3) {
            MS1 = ms1;
            MS2 = ms2;
            MS3 = ms3;
            pinMode(MS1, OUTPUT);
            pinMode(MS2, OUTPUT);
            pinMode(MS3, OUTPUT);
            are_MS_pins_set = true;
            return 0;
        }

        int set_microstep_mode(int ms) {
            if (are_pins_set == false) {return -1;} //check if pins are set
            micro_step = ms;
            scale_factor = pow(2, micro_step);
            
            if (are_MS_pins_set == true) {
                digitalWrite(MS1, (int) _resolution[micro_step][0]);
                digitalWrite(MS2, (int) _resolution[micro_step][1]);
                digitalWrite(MS3, (int) _resolution[micro_step][2]);
            }
            is_MS_set = true;
            return 0;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        int set_half_period_and_delay(float half_period, float time_delay_ms) {
            //Sets the half period and the time delay in each half period

            if (are_pins_set == false) {return -1;}
            if (is_MS_set == false) {return -1;}
            
            T_HP = half_period;
            t_delay = time_delay_ms;
            T_HP_nd = T_HP - t_delay/1000.0;
            return 0;
        }

        int set_target_pos(float angle) {
            //Sets the target position of the motor, given an angle in degrees
            
            if (are_pins_set == false) {return -1;}
            if (is_MS_set == false) {return -1;}
            
            float frac = abs(angle) / 360.0;
            unsigned int step_count = (unsigned int) (_SPR * scale_factor * frac);
            theta = step_count;
            theta_og = theta;

            theta_reached = false;
            t_at_reached = 0.0;

            omega = (float)theta / T_HP_nd; //set the angular velocity
            return 0;
        }

        int start() {
            //Gives the OK to start rotating motor
            
            //Set the target position and speed in AccelStepper object
            //Note: setSpeed must be after moveTo, because moveTo recalculates the velocity values
            AccelStepperObject.moveTo(theta);
            AccelStepperObject.setSpeed(omega);
            return 0;
        }

        void loop_check() {
            //Call this function in main loop
            
            if ((AccelStepperObject.distanceToGo() == 0) || (theta_reached == true)) {
                if (theta_reached == false) {
                    theta_reached = true;
                    t_at_reached = millis();
                } else {
                    if ((millis() - t_at_reached) > t_delay) {
                        omega = -1 * omega;
                        if (theta == theta_og) {theta = 0;}
                        else {theta = theta_og;}
                        AccelStepperObject.moveTo(theta);
                        AccelStepperObject.setSpeed(omega);
                        theta_reached = false;
                    }
                }
            }
        }

    private:
        const int _resolution[5][3];
        int _SPR; //steps per revolution in full step mode
        int motorInterfaceType;
        
        float T_HP; //total half period [s], including the delay time
        float T_HP_nd; //half period [s], not including the delay time
        float t_delay; //delay time after each T_HP_nd [ms]
        //unsigned int theta; //target position in steps
        unsigned int theta_og; //target position in steps, constant which will not be changed
        //float omega; //stepper velocity [steps/s]

        boolean theta_reached;
        long t_at_reached;
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const int dirPin1 = 2;
int stepPin1 = 3;
const int sleepreset1 = 4;
const int dirPin2 = 5;
const int stepPin2 = 6;
const int sleepreset2 = 7;
const int MS3 = 10;
const int MS2 = 9;
const int MS1 = 8;
const int micro_step_mode = 0;

int button_state = 0;
const int button_pin = A0;
const int relayPin = 11;

const float t_delay_1 = 50; //delay in [ms]
const float t_delay_2 = 100; //delay in [ms]
float pos_1 = 180.0; //stepper 1 target position [degrees]
float pos_2 = 360.0; //stepper 2 target position [degrees]
const float T_HP_1 = 0.3; //stepper 1 half-period [s]
const float T_HP_2 = 1.0; //stepper 2 half-period [s]

StepperConstSpeedSwitch stepper1 = StepperConstSpeedSwitch(dirPin1, stepPin1, sleepreset1);
StepperConstSpeedSwitch stepper2 = StepperConstSpeedSwitch(dirPin2, stepPin2, sleepreset2);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void setup() {
    Serial.begin(9600);
    Serial.println("\nHello world");
    Serial.println("Arduino ready");

    stepper1.set_pins();
    stepper1.set_microstep_pins(MS1, MS2, MS3);
    stepper1.set_microstep_mode(micro_step_mode);
    stepper1.set_half_period_and_delay(T_HP_1, t_delay_1);
    stepper1.set_target_pos(pos_1);

    stepper2.set_pins();
    stepper2.set_microstep_mode(micro_step_mode);
    stepper2.set_half_period_and_delay(T_HP_2, t_delay_2);
    stepper2.set_target_pos(pos_2);
    
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

    stepper1.start();
    stepper2.start();
}
 
void loop() {
    stepper1.loop_check();
    stepper2.loop_check();
    stepper1.AccelStepperObject.runSpeedToPosition();
    stepper2.AccelStepperObject.runSpeedToPosition();
}
