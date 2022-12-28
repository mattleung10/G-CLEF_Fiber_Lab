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
int delayMS = 200000; // delay between steps in microseconds


void setup() {
    // put your setup code here, to run once:
    pinMode(dirPin, OUTPUT);
    pinMode(stepPin, OUTPUT);
    pinMode(sleepreset, OUTPUT);
    pinMode(MS1, OUTPUT);
    pinMode(MS2, OUTPUT);
    pinMode(MS3, OUTPUT);

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
