#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <ArduinoBLE.h>

// ── BLE ───────────────────────────────────────────────────────────────────────
BLEService imuService("12345678-1234-1234-1234-123456789abc");
BLECharacteristic imuChar(
    "87654321-4321-4321-4321-cba987654321",
    BLENotify,
    44,
    true
);

// ── IMU ───────────────────────────────────────────────────────────────────────
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28, &Wire);

// ── Timing ───────────────────────────────────────────────────────────────────
#define SAMPLE_RATE_HZ  50
#define SAMPLE_INTERVAL (1000 / SAMPLE_RATE_HZ)   // 20ms
unsigned long lastSample = 0;

// ── Sleep/Wake ────────────────────────────────────────────────────────────────
void goToSleep() {
    bno.enterSuspendMode();
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDB, HIGH);
    Serial.println("Sleeping — waiting for BLE connection...");
}

void wakeUp() {
    bno.enterNormalMode();
    delay(500);
    Serial.println("Waking up — BNO055 resuming...");
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(2000);

    pinMode(LEDR, OUTPUT); digitalWrite(LEDR, HIGH);
    pinMode(LEDG, OUTPUT); digitalWrite(LEDG, HIGH);
    pinMode(LEDB, OUTPUT); digitalWrite(LEDB, HIGH);

    Serial.println("=== WearableBLE Starting ===");
    Serial.println("Initialising BNO055...");

    if (!bno.begin()) {
        Serial.println("BNO055 not detected — check wiring!");
        digitalWrite(LEDR, LOW);
        while (1);
    }
    bno.setExtCrystalUse(true);
    Serial.println("BNO055 ready!");

    Serial.println("Initialising BLE...");
    if (!BLE.begin()) {
        Serial.println("BLE failed!");
        digitalWrite(LEDR, LOW);
        while (1);
    }

    BLE.setLocalName("WearableBLE");
    BLE.setAdvertisedService(imuService);
    imuService.addCharacteristic(imuChar);
    BLE.addService(imuService);
    BLE.advertise();

    Serial.println("Advertising as Device 1 — ready!");
    digitalWrite(LEDB, LOW);   // blue = advertising
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
    BLEDevice central = BLE.central();

    if (central) {
        wakeUp();
        Serial.print("Connected: ");
        Serial.println(central.address());
        digitalWrite(LEDG, LOW);    // green = connected
        digitalWrite(LEDB, HIGH);

        while (central.connected()) {
            unsigned long now = millis();
            if (now - lastSample >= SAMPLE_INTERVAL) {
                lastSample = now;
                sendImuPacket();
            }
        }

        Serial.println("Disconnected");
        digitalWrite(LEDG, HIGH);
        goToSleep();

    } else {
        // Slow pulse blue — sleeping/advertising
        static unsigned long lastPulse = 0;
        unsigned long now = millis();
        if (now - lastPulse > 1000) {
            lastPulse = now;
            digitalWrite(LEDB, LOW);
            delay(100);
            digitalWrite(LEDB, HIGH);
        }
    }
}

// ── Send Packet ───────────────────────────────────────────────────────────────
// 44 bytes: timestamp(4) + accel xyz(12) + gyro xyz(12) + quat wxyz(16)
void sendImuPacket() {
    imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
    imu::Vector<3> gyro  = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    imu::Quaternion quat = bno.getQuat();

    uint8_t packet[44];
    uint32_t ts = millis();

    float ax = (float)accel.x();
    float ay = (float)accel.y();
    float az = (float)accel.z();
    float gx = (float)gyro.x();
    float gy = (float)gyro.y();
    float gz = (float)gyro.z();
    float qw = (float)quat.w();
    float qx = (float)quat.x();
    float qy = (float)quat.y();
    float qz = (float)quat.z();

    memcpy(packet+0,  &ts, 4);
    memcpy(packet+4,  &ax, 4);
    memcpy(packet+8,  &ay, 4);
    memcpy(packet+12, &az, 4);
    memcpy(packet+16, &gx, 4);
    memcpy(packet+20, &gy, 4);
    memcpy(packet+24, &gz, 4);
    memcpy(packet+28, &qw, 4);
    memcpy(packet+32, &qx, 4);
    memcpy(packet+36, &qy, 4);
    memcpy(packet+40, &qz, 4);

    imuChar.writeValue(packet, 44);

    // Debug
    Serial.print("ax="); Serial.print(ax,3);
    Serial.print(" ay="); Serial.print(ay,3);
    Serial.print(" az="); Serial.print(az,3);
    Serial.print(" | gx="); Serial.print(gx,2);
    Serial.print(" gy="); Serial.print(gy,2);
    Serial.print(" gz="); Serial.print(gz,2);
    Serial.print(" | qw="); Serial.print(qw,3);
    Serial.print(" qx="); Serial.print(qx,3);
    Serial.print(" qy="); Serial.print(qy,3);
    Serial.print(" qz="); Serial.println(qz,3);
}
