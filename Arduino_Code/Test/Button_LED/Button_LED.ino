// May 9, 2022
// LED Button Test
// button_pin is an input, which is hooked up to GND via a 10kOhm resistor
// We have a push button, in which the two terminals are connected to 5V and button_pin
// When the button is pushed, the builtin LED will turn ON.

int button_state = 0;
const int button_pin = A0;

void setup() {
    Serial.begin(9600);
    Serial.println("Arduino Ready");
    pinMode(button_pin, INPUT);
    pinMode(LED_BUILTIN, OUTPUT); //You need this line! Otherwise, LED will behave weirdly, will fade etc.
}

void loop() {
    button_state = digitalRead(button_pin);
    if (button_state == HIGH) {
        digitalWrite(LED_BUILTIN, HIGH);
        Serial.println("ON");
    } else {
        digitalWrite(LED_BUILTIN, LOW);
        Serial.println("OFF");
    }
    delay(100);
}
