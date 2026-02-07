#include <Wire.h>

static const uint8_t MPU_ADDR = 0x68;

// MPU6050 registers
static const uint8_t REG_PWR_MGMT_1   = 0x6B;
static const uint8_t REG_ACCEL_CONFIG = 0x1C;
static const uint8_t REG_CONFIG       = 0x1A;
static const uint8_t REG_ACCEL_ZOUT_H = 0x3F;

// Physical constants
static const float G = 9.80665f;
static const float ACC_LSB_PER_G = 16384.0f; // ±2g
static const float ACC_SCALE = G / ACC_LSB_PER_G;

// Sampling
static const uint16_t FS_HZ = 250;
static const uint32_t DT_US = 1000000UL / FS_HZ;

long az0 = 0;
uint32_t sample_id = 0;
uint32_t t_prev_us = 0;

// ---------------- low-level I2C ----------------
static void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission(true);
}

static bool readAccelZ(int16_t &az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_ZOUT_H);
  if (Wire.endTransmission(false) != 0) return false;

  if (Wire.requestFrom(MPU_ADDR, (uint8_t)2) != 2) return false;

  az = (int16_t)((Wire.read() << 8) | Wire.read());
  return true;
}

// ---------------- calibration ----------------
static void calibrateZ() {
  long sum = 0;
  int cnt = 0;

  for (int i = 0; i < 400; i++) {
    int16_t az;
    if (readAccelZ(az)) {
      sum += az;
      cnt++;
    }
    delay(5);
  }
  if (cnt > 0) az0 = sum / cnt;
}

// ---------------- setup ----------------
void setup() {
  sample_id = 0;
  Serial.begin(115200);
  delay(800);

  Wire.begin();
  Wire.setClock(100000);

  // Wake up MPU6050
  writeReg(REG_PWR_MGMT_1, 0x00);
  delay(100);

  // ±2g
  writeReg(REG_ACCEL_CONFIG, 0x00);

  // DLPF = ~94 Hz (CONFIG = 0x02)
  writeReg(REG_CONFIG, 0x02);

  delay(200);

  int16_t test;
  if (!readAccelZ(test)) {
    Serial.println("MPU6050 ERROR");
    while (1);
  }

  calibrateZ();

  Serial.println("sample,time_s,az_ms2");

  t_prev_us = micros();
}

// ---------------- loop ----------------
void loop() {
  uint32_t now = micros();
  if (now - t_prev_us >= DT_US) {
    t_prev_us += DT_US;

    int16_t az_raw;
    if (readAccelZ(az_raw)) {
      float az = (az_raw - az0) * ACC_SCALE;
      float t = sample_id * (1.0f / FS_HZ);

      Serial.print(sample_id);
      Serial.print(',');
      Serial.print(t, 6);
      Serial.print(',');
      Serial.println(az, 6);

      sample_id++;
    }
  }
}
