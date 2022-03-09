/*
  Arduino BLE peripheral
  
  The circuit:
  - Arduino Nano 33 BLE
*/

#include <ArduinoBLE.h>
#include "LSM9DS1_m.h" // For IMU instance (NANO33 BLE)

//----------------------------------------------------------------------------------------------------------------------
// BLE UUIDs
//----------------------------------------------------------------------------------------------------------------------

#define BLE_UUID_BLECOMM_SERVICE                  "991F0000-7774-3332-15F5-90324778E1BF"
#define BLE_UUID_BLECOMM                          "991F0001-7774-3332-15F5-90324778E1BF"

//----------------------------------------------------------------------------------------------------------------------
// BLE
//----------------------------------------------------------------------------------------------------------------------

#define NUMBER_OF_SAMPLES   10*3

union data_u
{
  struct __attribute__( ( packed ) )
  {
    uint16_t values[NUMBER_OF_SAMPLES];
    uint8_t packetCounter = 0;
  };
  uint8_t bytes[ NUMBER_OF_SAMPLES * sizeof( uint16_t ) + sizeof packetCounter ];
};

typedef struct
{
  data_u data;
  uint32_t index = 0;
  bool updated = false;
} data_t;

data_t SensorData;

BLEService blecommService( BLE_UUID_BLECOMM_SERVICE );
BLECharacteristic blecommCharacteristic( BLE_UUID_BLECOMM, BLERead | BLENotify, sizeof SensorData.data.bytes );

#define BLE_DEVICE_NAME                           "KIMe BLEComm"
#define BLE_LOCAL_NAME                            "KIMe BLEComm"

//----------------------------------------------------------------------------------------------------------------------
// APP & I/O
//----------------------------------------------------------------------------------------------------------------------
#ifndef LED_BLUE_PIN
  #define LED_BLUE_PIN          D7
#endif

#ifndef LED_RED_PIN
  #define LED_RED_PIN           D8
#endif
   
#define BLE_LED_PIN LED_BUILTIN
void setup()
{
  Serial.begin(115200);
  while (!Serial);

  pinMode(BLE_LED_PIN, OUTPUT); // initialize the built-in LED pin to indicate when a central is connected
  digitalWrite(BLE_LED_PIN, LOW);

  // IMU(Accelerometer) Setting
  /* IMU setup for LSM9DS1*/
  /* default setup has all sensors active in continous mode. Sample rates
   *  are as follows: accelerationSampleRate = 109Hz 
   */
  if (!IMU.begin())
  {
    /* Something went wrong... Put this thread to sleep indefinetely. */
    Serial.println(F("Can't set IMU. Initialization has been failed."));
    while(1);
    return;
  }
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println("Hz");
  Serial.println("Acceleration(X, Y, Z) in Digit Value is ");
  Serial.print("G Scale: ");Serial.println(LSM9DS1_ACCEL_SCALE_G);
  Serial.print("Denominator Factor: ");Serial.println(LSM9DS1_ACCEL_FACTOR);

  // BLE Setting
  if ( !setupBleMode() )
  {
    Serial.println( "Failed to initialize BLE!" );
    while ( 1 );
  }
  else
  {
    Serial.println( "BLE initialized. Waiting for clients to connect." );
  }
}


void loop()
{
  bleTask();
  if ( sensorTask() )
  {
    printTask();
  }
}


bool sensorTask()
{
  const uint32_t SENSOR_UPDATE_INTERVAL = 1; //ms
  static uint32_t previousMillis = 0;
  static int16_t xyz[3];
  
  // Read IMU(Acceleration) Data
  if (IMU.accelerationAvailable()) {
    if (!IMU.readAcceleration(xyz)) {
      xyz[0] = NAN; xyz[1] = NAN; xyz[2] = NAN;
    }    
  }

  // Sensor Update Timing
  uint32_t currentMillis = millis();
  if ( currentMillis - previousMillis < SENSOR_UPDATE_INTERVAL )
  {
    return false;
  }
  previousMillis = currentMillis;


  // Update Sensor Data
  SensorData.data.values[SensorData.index++] = xyz[0]; // X
  SensorData.data.values[SensorData.index++] = xyz[1]; // Y
  SensorData.data.values[SensorData.index++] = xyz[2]; // Z
  SensorData.index = ( SensorData.index + 1 ) % NUMBER_OF_SAMPLES;
  if ( SensorData.index != 0 ) //Print Data
  {
    return false;
  }

  SensorData.updated = true;
  return true;
}


void printTask()
{
  Serial.print( SensorData.data.packetCounter );
  Serial.print( "\t" );

  for ( uint32_t i = 0; i < NUMBER_OF_SAMPLES; i++ )
  {
    Serial.print( SensorData.data.values[i] );
    if ( i < NUMBER_OF_SAMPLES - 1 )
    {
      Serial.print( "\t" );
    }
  }
  Serial.println();
}


bool setupBleMode()
{
  if ( !BLE.begin() )
  {
    return false;
  }

  // set advertised local name and service UUID
  BLE.setDeviceName( BLE_DEVICE_NAME );
  BLE.setLocalName( BLE_LOCAL_NAME );
  BLE.setAdvertisedService( blecommService );

  // BLE add characteristics
  blecommService.addCharacteristic( blecommCharacteristic );

  // add service
  BLE.addService( blecommService );

  // set the initial value for the characeristic
  blecommCharacteristic.writeValue( SensorData.data.bytes, sizeof SensorData.data.bytes );

  // set BLE event handlers
  BLE.setEventHandler( BLEConnected, blePeripheralConnectHandler );
  BLE.setEventHandler( BLEDisconnected, blePeripheralDisconnectHandler );

  // start advertising
  BLE.advertise();

  return true;
}


void bleTask()
{
  const uint32_t BLE_UPDATE_INTERVAL = 10; //ms
  static uint32_t previousMillis = 0;

  uint32_t currentMillis = millis();
  if ( currentMillis - previousMillis >= BLE_UPDATE_INTERVAL )
  {
    previousMillis = currentMillis;
    BLE.poll();
  }

  if ( SensorData.updated )
  {
    blecommCharacteristic.writeValue( SensorData.data.bytes, sizeof SensorData.data.bytes );
    SensorData.data.packetCounter++;
    SensorData.updated = false;
  }
}

void blePeripheralConnectHandler( BLEDevice central )
{
  digitalWrite( BLE_LED_PIN, HIGH );
  Serial.print( F( "Connected to central: " ) );
  Serial.println( central.address() );
}


void blePeripheralDisconnectHandler( BLEDevice central )
{
  digitalWrite( BLE_LED_PIN, LOW );
  Serial.print( F( "Disconnected from central: " ) );
  Serial.println( central.address() );
}
