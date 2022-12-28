// Matthew Leung
// May 2022
// Single stepper motor test, with SLEEP and RESET pins combined

const int dirPin = 2;
const int stepPin = 3;
const int sleepreset = 4;
//const int reset = 5;
const int MS3 = 10;
const int MS2 = 9;
const int MS1 = 8;

float pos = 0.0;
const boolean esp = true;

int step_count = 0;
int delayMS = 400000; // delay between steps in microseconds

int button_state = 0;
const int button_pin = A0;
const int relayPin = 11;


void setup() {
    // put your setup code here, to run once:
    pinMode(dirPin, OUTPUT);
    pinMode(stepPin, OUTPUT);
    pinMode(sleepreset, OUTPUT);
    pinMode(MS1, OUTPUT);
    pinMode(MS2, OUTPUT);
    pinMode(MS3, OUTPUT);

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


    digitalWrite(sleepreset, HIGH); // LOW means Sleep is happening, Tie the Reset pin down with Sleep

    // full step
    digitalWrite(MS1, LOW);
    digitalWrite(MS2, LOW);
    digitalWrite(MS3, LOW);

    step_count = 100;
}

void loop() {
    // put your main code here, to run repeatedly:
    digitalWrite(dirPin, HIGH); // HIGH for CW, LOW for CCW

    for (int i = 0; i < step_count; i++) {
        digitalWrite(stepPin, HIGH);
        digitalWrite(stepPin, LOW);
        delayMicroseconds(delayMS);
    }

    delay(500);

    digitalWrite(dirPin, LOW); // HIGH for CW, LOW for CCW

    for (int i = 0; i < step_count; i++) {
        digitalWrite(stepPin, HIGH);
        digitalWrite(stepPin, LOW);
        delayMicroseconds(delayMS);
    }
    delay(500);
}
