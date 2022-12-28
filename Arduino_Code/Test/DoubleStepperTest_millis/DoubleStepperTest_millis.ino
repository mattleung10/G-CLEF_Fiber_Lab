// Control two motors, with step delay by millis
// Basically, if a certain appropriate amount of time has elapsed between steps, take the next step.

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

int stepper1speed = 200 * scale_factor; //steps per second
int stepper2speed = 800 * scale_factor; //steps per second
int stepper1targetpos = 100 * scale_factor; //position in steps
int stepper2targetpos = 150 * scale_factor; //position in steps

unsigned long stepper1PrevTime = millis();
unsigned long stepper2PrevTime = millis();
unsigned long stepper1CurrentTime = 0.0;
unsigned long stepper2CurrentTime = 0.0;

long stepper1Interval = 1.0/stepper1speed * 1000;
long stepper2Interval = 1.0/stepper2speed * 1000;

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

    pinMode(dirPin1, OUTPUT);
    pinMode(dirPin2, OUTPUT);
    pinMode(stepPin1, OUTPUT);
    pinMode(stepPin2, OUTPUT);

    digitalWrite(dirPin1, HIGH);
    digitalWrite(dirPin2, HIGH);
    digitalWrite(stepPin1, LOW);
    digitalWrite(stepPin2, LOW);
}

void stepper1step() {
    stepper1CurrentTime = millis();
    digitalWrite(stepPin1, LOW);
    
    if(stepper1CurrentTime - stepper1PrevTime > stepper1Interval){
        digitalWrite(stepPin1, HIGH);
        digitalWrite(stepPin1, LOW);
        stepper1PrevTime = stepper1CurrentTime;
    }
}

void stepper2step() {
    stepper2CurrentTime = millis();
    if(stepper2CurrentTime - stepper2PrevTime > stepper2Interval){
        digitalWrite(stepPin2, HIGH);
        digitalWrite(stepPin2, LOW);
        stepper2PrevTime = stepper2CurrentTime;
    }
}

void loop() {
    stepper1step();
    stepper2step();
}
