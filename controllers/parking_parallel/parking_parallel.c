/*
 * File:          parking_parallel.c
 * Date:          07/03/2025
 * Description: autonomous car parking simulator with RRT*-based path planning 
 * Author: Yaraslau Yakubouski
 * Modifications: constantly
 */

/*
 * You may need to add include files like <webots/distance_sensor.h> or
 * <webots/motor.h>, etc.
 */
 
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <webots/vehicle/driver.h>
#include <webots/camera.h>
#include <webots/device.h>
#include <webots/gps.h>
#include <webots/keyboard.h>
#include <webots/lidar.h>
#include <webots/robot.h>
#include <webots/distance_sensor.h>

void wbu_driver_set_steering_angle();
/*
 * You may want to add macros here.
 */
#define TIME_STEP 64
#define UNKNOWN 99999.99
#define NUM_DIST_SENSORS 12
#define NUM_CAMERAS 7

// structures 

typedef struct {
  double x;
  double y;
  double theta;
} CarConfig;

typedef struct Node {
  CarConfig config;
  struct Node* parent;
} Node;

#define MAX_NODES 1000
typedef struct {
  CarConfig nodes[MAX_NODES];
  int count;
} Path;

typedef struct {
  int camera_width;
  int camera_height;
  double camera_fov;
} Camera;


// devices 

bool parking_search = false;
bool has_gps = false;
bool has_camera = false;
bool has_distance_sensors = false;
bool autopark=false;

// camera
WbDeviceTag cameras[NUM_CAMERAS];

int camera_width[NUM_CAMERAS];
int camera_height[NUM_CAMERAS];
double camera_fov[NUM_CAMERAS];

// SICK laser
WbDeviceTag sick;
int sick_width = -1;
double sick_range = -1.0;
double sick_fov = -1.0;

// gps
WbDeviceTag gps;
double gps_coords[3] = {0.0, 0.0, 0.0};
double gps_speed = 0.0;

// distance sensors
WbDeviceTag distance_sensors[NUM_DIST_SENSORS]; 

// Line following PID
#define KP 0.25
#define KI 0.006
#define KD 2
#define KAW 0.1
double previous_error =0.0;
double integral=0.0;
double derivative=0.0;
double previous_steering_angle = 0.0;
// variables 

// misc
double speed = 0.0;
double steering_angle = 0.0;
int manual_steering = 0;
bool autodrive = false;

void set_speed(double kmh) {
  // max speed
  if (kmh > 250.0)
    kmh = 250.0;

  speed = kmh;

  //printf("setting speed to %g km/h\n", kmh);
  fflush(stdout);
  wbu_driver_set_cruising_speed(kmh);
}

// positive: turn right, negative: turn left
void set_steering_angle(double wheel_angle) {
  // limit the difference with previous steering_angle
  if (wheel_angle - steering_angle > 0.1)
    wheel_angle = steering_angle + 0.1;
  if (wheel_angle - steering_angle < -0.1)
    wheel_angle = steering_angle - 0.1;
  steering_angle = wheel_angle;
  // limit range of the steering angle
  if (wheel_angle > 0.5)
    wheel_angle = 0.5;
  else if (wheel_angle < -0.5)
    wheel_angle = -0.5;
  wbu_driver_set_steering_angle(wheel_angle);
}

void change_manual_steer_angle(int inc) {
  // set_autodrive(false);

  double new_manual_steering = manual_steering + inc;
  if (new_manual_steering <= 25.0 && new_manual_steering >= -25.0) {
    manual_steering = new_manual_steering;
    set_steering_angle(manual_steering * 0.02);
  }
  /* 
  if (manual_steering == 0)
    printf("going straight\n");
  else
    printf("turning %.2f rad (%s)\n", steering_angle, steering_angle < 0 ? "left" : "right");
  fflush(stdout);
  */
}

void check_keyboard() {
  int key = wb_keyboard_get_key();
  switch (key) {
    case WB_KEYBOARD_UP:
      set_speed(speed + 0.5);
      break;
    case WB_KEYBOARD_DOWN:
      set_speed(speed - 0.5);
      break;
    case WB_KEYBOARD_RIGHT:
      change_manual_steer_angle(+2);
      break;
    case WB_KEYBOARD_LEFT:
      change_manual_steer_angle(-2);
      break;
  }
}

void print_help() {
  printf("You can drive this car!\n");
  printf("Select the 3D window and then use the cursor keys to:\n");
  printf("[LEFT]/[RIGHT] - steer\n");
  printf("[UP]/[DOWN] - accelerate/slow down\n");
}

void compute_gps_speed() {
  const double *coords = wb_gps_get_values(gps);
  const double speed_ms = wb_gps_get_speed(gps);
  // store into global variables
  gps_speed = speed_ms * 3.6;  // convert from m/s to km/h
  memcpy(gps_coords, coords, sizeof(gps_coords));
}
/*
double process_camera_image(const unsigned char *sick_data)
{
  //int num_pixels = camera_height * camera_width;
  ;
}
*/
double* process_distance_sensors(WbDeviceTag *sensors, int num_sensors) {
  double *distances = (double *)calloc(num_sensors,sizeof(double));
  for (int i = 0; i < num_sensors - 1; i++) {
    distances[i] = wb_distance_sensor_get_value(sensors[i]);
    //if (i == num_sensors-1) printf("%0.2f \n",distances[i]);
    //else printf("%0.2f ",distances[i]);
  }
  return distances;
}

double compute_planner_steering_angle(CarConfig current, CarConfig next) {
    // compute desired heading angle to the target point
    double theta_desired = atan2(next.y - current.y, next.x - current.x);
    
    // compute error (difference between desired and current heading)
    double error = theta_desired - current.theta;

    //while (error > M_PI) error -= 2 * M_PI;
    //while (error < -M_PI) error += 2 * M_PI;

    // compute integral and derivative terms
    integral += error;                              // Accumulate integral
    double derivative = error - previous_error;     // Compute derivative

    // compute raw steering angle
    double steering_angle = KP * error + KI * integral + KD * derivative;

    // steering angle saturation
    double clamped_steering_angle = steering_angle;
    if (clamped_steering_angle > 25.0)
        clamped_steering_angle = 25.0;
    else if (clamped_steering_angle < -25.0)
        clamped_steering_angle = -25.0;

    // anti-windup: back-calculation
    double anti_windup = KAW * (clamped_steering_angle - steering_angle);
    integral += anti_windup;
    
    steering_angle = KP * error + (KI*error - anti_windup) * integral + KD * derivative;
    // store values for next iteration
    previous_error = error;
    previous_steering_angle = steering_angle;

    return steering_angle;
}

void perform_parking(int timestep)
{
  
}
/*
 * This is the main program.
 * The arguments of the main function can be specified by the
 * "controllerArgs" field of the Robot node
 */
int main(int argc, char **argv) {
  /* necessary to initialize webots stuff */
  wbu_driver_init();

  
  /*
   * You should declare here WbDeviceTag variables for storing
   * robot devices like this:
   *  WbDeviceTag my_sensor = wb_robot_get_device("my_sensor");
   *  WbDeviceTag my_actuator = wb_robot_get_device("my_actuator");
   */
  
  int j = 0;
  for (j = 0; j < wb_robot_get_number_of_devices(); ++j) {
    WbDeviceTag device = wb_robot_get_device_by_index(j);
    const char *name = wb_device_get_name(device);
    if (strcmp(name, "lidar(1)") == 0)
    {
      parking_search = true;
    }
    else if (strcmp(name, "gps") == 0)
      has_gps = true;
  }
  
  const char *sensor_names[NUM_DIST_SENSORS] = {
    "distance sensor left front side",
    "distance sensor front left",
    "distance sensor front lefter",
    "distance sensor front righter",
    "distance sensor front right",
    "distance sensor right front side",
    "distance sensor left side",
    "distance sensor left",
    "distance sensor lefter",
    "distance sensor righter",
    "distance sensor right",
    "distance sensor right side",
  };

    for (int i = 0; i < NUM_DIST_SENSORS; i++) {
      distance_sensors[i] = wb_robot_get_device(sensor_names[i]);
      if (distance_sensors[i]!=0){
      wb_distance_sensor_enable(distance_sensors[i], TIME_STEP);
      printf("found sensor\n");
      } else printf("sensor not found\n");
  }
  
  const char *camera_names[NUM_CAMERAS] = {
    "camera_front_top",
    "camera_front_bumper_wide",
    "camera_right_fender",
    "camera_right_pillar",
    "camera_left_fender",
    "camera_left_pillar",
    "camera_rear"
  };
  // camera devices
    for (int i = 0; i < NUM_CAMERAS; i++){
      cameras[i] = wb_robot_get_device(camera_names[i]);
      wb_camera_enable(cameras[i], TIME_STEP);
      camera_width[i] = wb_camera_get_width(cameras[i]);
      camera_height[i] = wb_camera_get_height(cameras[i]);
      camera_fov[i] = wb_camera_get_fov(cameras[i]);
      if (wb_camera_has_recognition(cameras[i]))
      {
        wb_camera_recognition_enable(cameras[i],TIME_STEP);
        wb_camera_recognition_enable_segmentation(cameras[i]);
      }
    }
    /*
  // SICK sensor
  if (parking_search) {
    sick = wb_robot_get_device("lidar(1)");
    wb_lidar_enable(sick, TIME_STEP);
    wb_lidar_enable_point_cloud(sick);
    sick_width = wb_lidar_get_horizontal_resolution(sick);
    sick_range = wb_lidar_get_max_range(sick);
    sick_fov = wb_lidar_get_fov(sick);
  }
    */
  // initialize gps
  if (has_gps) {
    gps = wb_robot_get_device("gps");
    wb_gps_enable(gps, TIME_STEP);
  }

  print_help();
  
  // switching manual/automatic

  wb_keyboard_enable(TIME_STEP);
  /* main loop
   * Perform simulation steps of TIME_STEP milliseconds
   * and leave the loop when the simulation is over
   */
  while (wb_robot_step(TIME_STEP) != -1) {

    
    /*
     * Read the sensors :
     * Enter here functions to read sensor data, like:
     *  double val = wb_distance_sensor_get_value(my_sensor);
     */
    fflush(stdout);
    /* Process sensor data here */
      if (has_gps) compute_gps_speed();
      double *values = process_distance_sensors(distance_sensors, NUM_DIST_SENSORS); // pointer to array for storing distances
      for (int k = 0; k < NUM_DIST_SENSORS; k++){
      printf("%0.2f ",values[k]);
      if (k==NUM_DIST_SENSORS-1) printf("\n");
      if (k==NUM_DIST_SENSORS/2-1) printf("| ");
      fflush(stdout);
      }
      //const WbLidarPoint *sick_data = wb_lidar_get_point_cloud(sick);
      
      // check if user pressed 'P' (for parking)
      /*
      int key = wb_keyboard_get_key();
      if (key == 'P' || key == 'p') 
      {
        double current_speed=wbu_driver_get_current_speed();
        if (current_speed!=0.0 )
          printf("Please stop the vehicle for the sequence to start\n");
        else{
          for (int j=0;j<10;j++) wb_robot_step(TIME_STEP);
          perform_parking(TIME_STEP);
        }
      } 
      */
      check_keyboard();
      free(values);
    /*
     * Enter here functions to send actuator commands, like:
     * wb_motor_set_position(my_actuator, 10.0);
     */
      
  }

  /* Enter your cleanup code here */

  /* This is necessary to cleanup webots resources */
  wb_robot_cleanup();

  return 0;
}
