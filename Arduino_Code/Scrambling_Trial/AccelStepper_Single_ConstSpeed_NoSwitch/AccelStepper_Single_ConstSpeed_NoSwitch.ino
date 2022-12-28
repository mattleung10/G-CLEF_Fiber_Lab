// June 2022
// Using AccelStepper to drive motor
// With constant speed, no switching directions
// Repinned May 18, 2022

#include <AccelStepper.h>

class StepperConstSpeed {
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

        float nu;

        AccelStepper AccelStepperObject; 
        
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        StepperConstSpeed(int di, int st, int slre) :
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

        int set_frequency(float freq) {
            //Sets the frequency
            if (are_pins_set == false) {return -1;}
            if (is_MS_set == false) {return -1;}
            nu = freq;
            steps_per_s = nu * _SPR * scale_factor;
            AccelStepperObject.setSpeed(steps_per_s);
            return 0;
        }

        void run_loop() {
            //Call this function in the main loop
            AccelStepperObject.runSpeed();
        }

    private:
        const int _resolution[5][3];
        int _SPR; //steps per revolution in full step mode
        int motorInterfaceType;
        
        float steps_per_s;
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
const int micro_step_mode = 2;

int button_state = 0;
const int button_pin = A0;
const int relayPin = 11;

//The frequency for the motors. Use negative numbers to go the opposite direction.
const float freq_1 = 1; //stepper 1 frequency in [Hz]
const float freq_2 = 1.5; //stepper 2 frequency in [Hz]

StepperConstSpeed stepper1 = StepperConstSpeed(dirPin1, stepPin1, sleepreset1);
//StepperConstSpeedSwitch stepper2 = StepperConstSpeed(dirPin2, stepPin2, sleepreset2);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void setup() {
    Serial.begin(9600);
    Serial.println("\nHello world");
    Serial.println("Arduino ready");

    stepper1.set_pins();
    stepper1.set_microstep_pins(MS1, MS2, MS3);
    stepper1.set_microstep_mode(micro_step_mode);
    stepper1.set_frequency(freq_1);

    //stepper2.set_pins();
    //stepper2.set_microstep_mode(micro_step_mode);
    //stepper2.set_frequency(freq_2);
    
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

}
 
void loop() {
    stepper1.run_loop();
    //stepper2.run_loop();
}
