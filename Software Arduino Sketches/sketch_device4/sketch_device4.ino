#include <bluefruit.h>
#include "LSM6DS3.h"
#include "Wire.h"
#include <MadgwickAHRS.h>

// ── Change per device the 3 lines below ──────────────────────────────────────────
#define DEVICE_NAME  "SensePlus_4"
#define SERVICE_UUID "45678901-4567-4567-4567-456789012345"
#define CHAR_UUID    "10987654-7654-7654-7654-fedcba987654"
// ─────────────────────────────────────────────────────────────────────────────

LSM6DS3 imu(I2C_MODE, 0x6A);
Madgwick filter;

BLEService        imuService(SERVICE_UUID);
BLECharacteristic imuChar(CHAR_UUID, CHR_PROPS_NOTIFY, 44);

#define SAMPLE_RATE_HZ  50
#define SAMPLE_INTERVAL (1000 / SAMPLE_RATE_HZ)
unsigned long lastSample = 0;

float gx_bias=0, gy_bias=0, gz_bias=0;
float ax_bias=0, ay_bias=0;
float ax, ay, az, gx, gy, gz;

void calibrateSensors() {
    Serial.println("Keep Flat till calibration...");
    delay(2000);
    float gxs=0, gys=0, gzs=0, axs=0, ays=0;
    for (int i=0; i<500; i++) {
        gxs += imu.readFloatGyroX();
        gys += imu.readFloatGyroY();
        gzs += imu.readFloatGyroZ();
        axs += imu.readFloatAccelX();
        ays += imu.readFloatAccelY();
        delay(4);
    }
    gx_bias=gxs/500; gy_bias=gys/500; gz_bias=gzs/500;
    ax_bias=axs/500; ay_bias=ays/500;
    Serial.println("Calibration done!");
    Serial.print("Gyro bias: ");
    Serial.print(gx_bias); Serial.print(", ");
    Serial.print(gy_bias); Serial.print(", ");
    Serial.println(gz_bias);
}

void connect_callback(uint16_t conn_handle) {
    BLEConnection* conn = Bluefruit.Connection(conn_handle);
    conn->requestMtuExchange(128);
    Serial.println("Connected — MTU exchange requested");
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_BLUE, HIGH);
}

void disconnect_callback(uint16_t conn_handle, uint8_t reason) {
    Serial.println("Disconnected — advertising again");
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_BLUE, LOW);
}

void setupBLE() {
    Bluefruit.configPrphConn(128, BLE_GAP_EVENT_LENGTH_DEFAULT, 1, 1);
    Bluefruit.begin();
    Bluefruit.setName(DEVICE_NAME);
    Bluefruit.Periph.setConnectCallback(connect_callback);
    Bluefruit.Periph.setDisconnectCallback(disconnect_callback);

    imuService.begin();
    imuChar.setProperties(CHR_PROPS_NOTIFY);
    imuChar.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
    imuChar.setFixedLen(44);
    imuChar.begin();

    Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
    Bluefruit.Advertising.addTxPower();
    Bluefruit.Advertising.addService(imuService);
    Bluefruit.ScanResponse.addName();
    Bluefruit.Advertising.restartOnDisconnect(true);
    Bluefruit.Advertising.setInterval(32, 244);
    Bluefruit.Advertising.start(0);

    Serial.print("BLE advertising as ");
    Serial.println(DEVICE_NAME);
}

void setup() {
    Serial.begin(115200);
    unsigned long t = millis();
    while (!Serial && millis()-t < 3000);
    delay(500);

    pinMode(LED_RED,   OUTPUT); digitalWrite(LED_RED,   HIGH);
    pinMode(LED_GREEN, OUTPUT); digitalWrite(LED_GREEN, HIGH);
    pinMode(LED_BLUE,  OUTPUT); digitalWrite(LED_BLUE,  HIGH);

    Serial.print("=== "); Serial.print(DEVICE_NAME); Serial.println(" Starting ===");
    Serial.println("Starting IMU...");

    imu.settings.gyroRange       = 2000;
    imu.settings.accelRange      = 8;
    imu.settings.gyroSampleRate  = 416;
    imu.settings.accelSampleRate = 416;

    if (imu.begin() != 0) {
        Serial.println("IMU ERROR!");
        digitalWrite(LED_RED, LOW);
        while(1);
    }
    Serial.println("IMU OK — ±8g accel / ±2000dps gyro");

    calibrateSensors();

    filter.begin(SAMPLE_RATE_HZ);
    filter.beta = 0.033f;

    setupBLE();
    digitalWrite(LED_BLUE, LOW);
}

void loop() {
    unsigned long now = millis();
    if (Bluefruit.connected()) {
        if (now - lastSample >= SAMPLE_INTERVAL) {
            lastSample = now;
            updateIMU();
            sendPacket();
        }
    } else {
        static unsigned long lastPulse = 0;
        if (now - lastPulse > 1000) {
            lastPulse = now;
            digitalWrite(LED_BLUE, LOW);
            delay(100);
            digitalWrite(LED_BLUE, HIGH);
        }
    }
}

void updateIMU() {
    // LSM6DS3 readFloat returns values in g, converting to m/s²
    ax = (imu.readFloatAccelX() - ax_bias) * 9.80665f;
    ay = (imu.readFloatAccelY() - ay_bias) * 9.80665f;
    az = (imu.readFloatAccelZ()) * 9.80665f;

    // Gyro stays in deg/s, no conversion here
    gx = imu.readFloatGyroX() - gx_bias;
    gy = imu.readFloatGyroY() - gy_bias;
    gz = imu.readFloatGyroZ() - gz_bias;

    filter.updateIMU(
        gx * DEG_TO_RAD,
        gy * DEG_TO_RAD,
        gz * DEG_TO_RAD,
        ax / 9.80665f,   // Madgwickfilter needs g units hence conversion from m/s² to g
        ay / 9.80665f,
        az / 9.80665f
    );
}

void sendPacket() {
    uint8_t packet[44];
    uint32_t ts = millis();

    float roll  = filter.getRoll();
    float pitch = filter.getPitch();
    float yaw   = filter.getYaw();

    // Impact detection 
// Now comparing in m/s² 
float mag = sqrt(ax*ax + ay*ay + az*az);
if (abs(mag - 9.81f) > 15.0f) {   // 15 m/s² above gravity = real impact for calibration
    Serial.print("Impact");
    Serial.print(mag);
    Serial.println(" m/s²");
}

    memcpy(packet+0,  &ts,    4);
    memcpy(packet+4,  &ax,    4);
    memcpy(packet+8,  &ay,    4);
    memcpy(packet+12, &az,    4);
    memcpy(packet+16, &gx,    4);
    memcpy(packet+20, &gy,    4);
    memcpy(packet+24, &gz,    4);
    memcpy(packet+28, &roll,  4);
    memcpy(packet+32, &pitch, 4);
    memcpy(packet+36, &yaw,   4);
    float zero = 0;
    memcpy(packet+40, &zero,  4);

    imuChar.notify(packet, 44);

    // Debug output
    Serial.print("ax="); Serial.print(ax,3);
    Serial.print(" ay="); Serial.print(ay,3);
    Serial.print(" az="); Serial.print(az,3);
    Serial.print(" | gx="); Serial.print(gx,2);
    Serial.print(" gy="); Serial.print(gy,2);
    Serial.print(" gz="); Serial.print(gz,2);
    Serial.print(" | R="); Serial.print(roll,1);
    Serial.print(" P="); Serial.print(pitch,1);
    Serial.print(" Y="); Serial.println(yaw,1);
}