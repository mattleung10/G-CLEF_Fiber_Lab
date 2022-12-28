// Matthew Leung
// May 2022
// Single stepper motor test
// Hold motor in constant position

const int dirPin = 2;
const int stepPin = 3;
const int sleep = 4;
const int reset = 5;
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
    pinMode(sleep, OUTPUT);
    pinMode(reset, OUTPUT);
    pinMode(MS1, OUTPUT);
    pinMode(MS2, OUTPUT);
    pinMode(MS3, OUTPUT);

    digitalWrite(sleep, HIGH); // LOW means Sleep is happening
    digitalWrite(reset, HIGH); // Tie the Reset pin down with Sleep

    // full step
    digitalWrite(MS1, LOW);
    digitalWrite(MS2, LOW);
    digitalWrite(MS3, LOW);
}

void loop() {
}
