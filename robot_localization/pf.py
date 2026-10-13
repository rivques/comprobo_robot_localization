#!/usr/bin/env python3

""" This is the starter code for the robot localization project """

import rclpy
from threading import Thread
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud
from nav2_msgs.msg import Particle as Nav2Particle
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from rclpy.duration import Duration
import math
import time
import numpy as np
from occupancy_field import OccupancyField
from helper_functions import TFHelper, draw_random_sample
from rclpy.qos import qos_profile_sensor_data
from angle_helpers import quaternion_from_euler

class Particle(object):
    """ Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)
        Attributes:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
        """ Construct a new Particle
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of KeyboardInterruptthe hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized """ 
        self.w = w
        self.theta = theta
        self.x = x
        self.y = y

    def as_pose(self):
        """ A helper function to convert a particle to a geometry_msgs/Pose message """
        q = quaternion_from_euler(0, 0, self.theta)
        return Pose(position=Point(x=self.x, y=self.y, z=0.0),
                    orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))

    def __repr__(self) -> str:
        return f"Particle(x={self.x:.3f}, y={self.y:.3f}, theta={self.theta:.2f}, w={self.w:.3f})"

    # TODO: define additional helper functions if needed

class ParticleFilter(Node):
    """ The class that represents a Particle Filter ROS Node
        Attributes list:
            base_frame: the name of the robot base coordinate frame (should be "base_footprint" for most robots)
            map_frame: the name of the map coordinate frame (should be "map" in most cases)
            odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
            scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
            n_particles: the number of particles in the filter
            d_thresh: the amount of linear movement before triggering a filter update
            a_thresh: the amount of angular movement before triggering a filter update
            pose_listener: a subscriber that listens for new approximate pose estimates (i.e. generated through the rviz GUI)
            particle_pub: a publisher for the particle cloud
            last_scan_timestamp: this is used to keep track of the clock when using bags
            scan_to_process: the scan that our run_loop should process next
            occupancy_field: this helper class allows you to query the map for distance to closest obstacle
            transform_helper: this helps with various transform operations (abstracting away the tf2 module)
            particle_cloud: a list of particles representing a probability distribution over robot poses
            current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                                   The pose is expressed as a list [x,y,theta] (where theta is the yaw)
            thread: this thread runs your main loop
    """
    def __init__(self):
        super().__init__('pf')
        self.base_frame = "base_footprint"   # the frame of the robot base
        self.map_frame = "map"          # the name of the map coordinate frame
        self.odom_frame = "odom"        # the name of the odometry coordinate frame
        self.scan_topic = "scan"        # the topic where we will get laser scans from 

        self.n_particles = 1300          # the number of particles to use

        self.d_thresh = 0.2             # the amount of linear movement before performing an update
        self.a_thresh = math.pi/6       # the amount of angular movement before performing an update

        # TODO: define additional constants if needed

        # pose_listener responds to selection of a new approximate robot location (for instance using rviz)
        self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.update_initial_pose, 10)

        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = self.create_publisher(ParticleCloud, "particle_cloud", qos_profile_sensor_data)

        # laser_subscriber listens for data from the lidar
        self.create_subscription(LaserScan, self.scan_topic, self.scan_received, 10)

        # this is used to keep track of the timestamps coming from bag files
        # knowing this information helps us set the timestamp of our map -> odom
        # transform correctly
        self.last_scan_timestamp = None
        # this is the current scan that our run_loop should process
        self.scan_to_process = None
        # your particle cloud will go here
        self.particle_cloud = []

        self.current_odom_xy_theta = []
        self.occupancy_field = OccupancyField(self, extra_meters = 15)
        self.transform_helper = TFHelper(self)

        # we are using a thread to work around single threaded execution bottleneck
        thread = Thread(target=self.loop_wrapper)
        thread.start()
        self.transform_update_timer = self.create_timer(0.05, self.pub_latest_transform)

    def pub_latest_transform(self):
        """ This function takes care of sending out the map to odom transform """
        if self.last_scan_timestamp is None:
            return
        postdated_timestamp = Time.from_msg(self.last_scan_timestamp) + Duration(seconds=0.1)
        self.transform_helper.send_last_map_to_odom_transform(self.map_frame, self.odom_frame, postdated_timestamp)

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        while True:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        """ This is the main run_loop of our particle filter.  It checks to see if
            any scans are ready and to be processed and will call several helper
            functions to complete the processing.
            
            You do not need to modify this function, but it is helpful to understand it.
        """
        if self.scan_to_process is None:
            return
        msg = self.scan_to_process

        (new_pose, delta_t) = self.transform_helper.get_matching_odom_pose(self.odom_frame,
                                                                           self.base_frame,
                                                                           msg.header.stamp)
        if new_pose is None:
            # we were unable to get the pose of the robot corresponding to the scan timestamp
            if delta_t is not None and delta_t < Duration(seconds=0.0):
                # we will never get this transform, since it is before our oldest one
                self.scan_to_process = None
            return
        
        (r, theta) = self.transform_helper.convert_scan_to_polar_in_robot_frame(msg, self.base_frame)
        # print("r[0]={0}, theta[0]={1}".format(r[0], theta[0]))
        # clear the current scan so that we can process the next one
        self.scan_to_process = None

        self.odom_pose = new_pose
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        # print("x: {0}, y: {1}, yaw: {2}".format(*new_odom_xy_theta))

        if not self.current_odom_xy_theta:
            self.current_odom_xy_theta = new_odom_xy_theta
        elif not self.particle_cloud:
            # now that we have all of the necessary transforms we can update the particle cloud
            self.initialize_particle_cloud(msg.header.stamp)
        elif self.moved_far_enough_to_update(new_odom_xy_theta):
            start_time = time.monotonic()
            # we have moved far enough to do an update!
            self.resample_particles()               # resample particles to focus on areas of high density
            self.update_particles_with_odom()    # update based on odometry
            self.update_particles_with_laser(r, theta)   # update based on laser scan
            self.update_robot_pose()                # update robot's pose based on particles
            end_time = time.monotonic()
            print(f"particle filter update took {end_time-start_time:.2f} real seconds")
        # publish particles (so things like rviz can see them)
        self.publish_particles(msg.header.stamp)

    def moved_far_enough_to_update(self, new_odom_xy_theta):
        return math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or \
               math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or \
               math.fabs(new_odom_xy_theta[2] - self.current_odom_xy_theta[2]) > self.a_thresh


    def update_robot_pose(self):
        """ Update the estimate of the robot's pose given the updated particles.
            There are two logical methods for this:
                (1): compute the mean pose
                (2): compute the most likely pose (i.e. the mode of the distribution)
        """
        # first make sure that the particle weights are normalized
        self.normalize_particles()

        # pick the most likely particle, that's our guess
        most_likely_particle = self.particle_cloud[np.argmax([particle.w for particle in self.particle_cloud])]
        self.robot_pose = most_likely_particle.as_pose()
        print(f"most likely particle has weight {most_likely_particle.w:.6f}")
        if hasattr(self, 'odom_pose'):
            self.transform_helper.fix_map_to_odom_transform(self.robot_pose,
                                                            self.odom_pose)
        else:
            self.get_logger().warn("Can't set map->odom transform since no odom data received")

    def update_particles_with_odom(self, ODOM_NOISE=0.30):
        """ Update the particles using the newly given odometry pose.
            Also, store the transform from the odom frame to the map frame
            in each particle.
            ODOM_NOISE: the amount of noise to add to the odometry update (as a fraction of the distance moved)
        """

        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        # compute the change in x,y,theta since our last update
        if self.current_odom_xy_theta:
            old_odom_xy_theta = self.current_odom_xy_theta
            delta = (new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                     new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                     new_odom_xy_theta[2] - self.current_odom_xy_theta[2])

            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            self.current_odom_xy_theta = new_odom_xy_theta
            return
        dist_translated = np.linalg.norm(delta[0:1])
        x_noise = np.random.normal(scale=ODOM_NOISE*dist_translated, size=len(self.particle_cloud))
        y_noise = np.random.normal(scale=ODOM_NOISE*dist_translated, size=len(self.particle_cloud))
        theta_noise = np.random.normal(scale=math.fabs(ODOM_NOISE*delta[2]), size=len(self.particle_cloud))

        # Based on the linear algebra from the conceptual overview:
        init_aff_odom = np.array([
            [np.cos(old_odom_xy_theta[2]), -np.sin(old_odom_xy_theta[2]), old_odom_xy_theta[0]],
            [np.sin(old_odom_xy_theta[2]),  np.cos(old_odom_xy_theta[2]), old_odom_xy_theta[1]],
            [0,                             0,                            1                   ]
        ])
        inv_init_aff_odom = np.linalg.inv(init_aff_odom)
        final_aff_odom = np.array([
            [np.cos(new_odom_xy_theta[2]), -np.sin(new_odom_xy_theta[2]), new_odom_xy_theta[0]],
            [np.sin(new_odom_xy_theta[2]),  np.cos(new_odom_xy_theta[2]), new_odom_xy_theta[1]],
            [0,                             0,                            1                   ]
        ])

        for i, particle in enumerate(self.particle_cloud):
            noisy_theta = particle.theta + theta_noise[i]
            init_particle_aff_map = np.array([
                [np.cos(noisy_theta), -np.sin(noisy_theta), particle.x + x_noise[i]],
                [np.sin(noisy_theta),  np.cos(noisy_theta), particle.y + y_noise[i]],
                [0,                       0,                      1                      ]
            ])

            transformation_odom_to_particle = np.matmul(init_particle_aff_map, inv_init_aff_odom)

            final_particle_aff_map = np.matmul(transformation_odom_to_particle, final_aff_odom)

            # extract the particle pose from the matrix
            particle.x = final_particle_aff_map[0, 2]
            particle.y = final_particle_aff_map[1, 2]
            particle.theta = np.atan2(final_particle_aff_map[1, 0], final_particle_aff_map[0, 0])

            # prevent these from moving off the map
            particle.x = np.clip(particle.x, self.occupancy_field.actual_origin_x_m, self.occupancy_field.actual_width_m + self.occupancy_field.actual_origin_x_m)
            particle.y = np.clip(particle.y, self.occupancy_field.actual_origin_y_m, self.occupancy_field.actual_height_m + self.occupancy_field.actual_origin_y_m)
            particle.theta = self.transform_helper.angle_normalize(particle.theta)

    def resample_particles(self):
        """ Resample the particles according to the new particle weights.
            The weights stored with each particle should define the probability that a particular
            particle is selected in the resampling step.  You may want to make use of the given helper
            function draw_random_sample in helper_functions.py.
        """

        proportion_uniform = 0.3

        # DONE: fill out the rest of the implementation
        self.particle_cloud = draw_random_sample(
            self.particle_cloud,
            [particle.w for particle in self.particle_cloud],
            int(self.n_particles * (1 - proportion_uniform))
        )

        particle_poses = np.random.uniform(
            (self.occupancy_field.actual_origin_x_m, self.occupancy_field.actual_origin_y_m, -np.pi),
            (self.occupancy_field.actual_width_m + self.occupancy_field.actual_origin_x_m, 
            self.occupancy_field.actual_height_m + self.occupancy_field.actual_origin_y_m, np.pi),
            (int(self.n_particles * proportion_uniform), 3)
        )
        self.particle_cloud += [Particle(x, y, theta) for x, y, theta in particle_poses]


    def update_particles_with_laser(self, r, theta):
        """ Updates the particle weights in response to the scan data
            r: the distance readings to obstacles, in meters
            theta: the angle relative to the robot frame for each corresponding reading, in radians
        """
        # TODO: implement this
        # polar to rectangular
        scan_points = np.array([
            [r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i]), 1]
            for i in range(len(r)) 
            if math.isfinite(r[i])
        ]).transpose()

        for particle in self.particle_cloud:
            T = np.array([
                [np.cos(particle.theta), -np.sin(particle.theta), particle.x],
                [np.sin(particle.theta), np.cos(particle.theta),  particle.y],
                [0,                      0,                       1         ]
            ])
            projected_points = np.matmul(T, scan_points)

            particle.w = 0
            added_points = 0

            for point in projected_points.T:
                obstacle_distance = self.occupancy_field.get_closest_obstacle_distance(point[0], point[1])
                if math.isfinite(obstacle_distance):
                    added_points += 1
                    particle.w += obstacle_distance ** 2

            particle.w /= added_points # normalize for number of points added

        print(f"Smallest particle weight is {min([particle.w for particle in self.particle_cloud]):.3}")

        # closest_scan_obstacle = min(r)
        # if not math.isfinite(closest_scan_obstacle):
        #     print(f"AAAAAAAA non-finite closest obstacle {closest_scan_obstacle} from scan {r}")
        #     return # don't try to update particles
        # bad_particles = []

        # for particle in self.particle_cloud:
        #     closest_obstacle = self.occupancy_field.get_closest_obstacle_distance(particle.x, particle.y)
        #     particle.w = abs(closest_obstacle - closest_scan_obstacle)
        #     if math.isnan(particle.w):
        #         print(f"AAAA NaN for particle {particle} and closest obstacle {closest_obstacle}")
        #         bad_particles.append(particle)

        # for bad_particle in bad_particles:
        #     self.particle_cloud.remove(bad_particle)
            
    def update_initial_pose(self, msg):
        """ Callback function to handle re-initializing the particle filter based on a pose estimate.
            These pose estimates could be generated by another ROS Node or could come from the rviz GUI """
        xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)
        self.initialize_particle_cloud(msg.header.stamp, xy_theta)

    def initialize_particle_cloud(self, timestamp, xy_theta=None):
        """ Initialize the particle cloud.
            Arguments
            xy_theta: a triple consisting of the mean x, y, and theta (yaw) to initialize the
                      particle cloud around.  If this input is omitted, the odometry will be used """
        if xy_theta is None:
            xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)

        # DONE create particles

        particle_poses = np.random.uniform(
            (self.occupancy_field.actual_origin_x_m, self.occupancy_field.actual_origin_y_m, -np.pi),
            (self.occupancy_field.actual_width_m + self.occupancy_field.actual_origin_x_m, 
            self.occupancy_field.actual_height_m + self.occupancy_field.actual_origin_y_m, np.pi),
            (self.n_particles, 3)
        )

        self.particle_cloud = [Particle(x, y, theta) for x, y, theta in particle_poses]

        self.normalize_particles()
        self.update_robot_pose()

    def normalize_particles(self):
        """ Make sure the particle weights define a valid distribution (i.e. sum to 1.0) """
        # DONE: implement this

        b = 0.4 # TODO tune this

        sum = 0

        for particle in self.particle_cloud:
            particle.w = b ** particle.w
            sum += particle.w

        for particle in self.particle_cloud:
            particle.w = particle.w / sum
        

    def publish_particles(self, timestamp):
        msg = ParticleCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = timestamp
        for p in self.particle_cloud:
            msg.particles.append(Nav2Particle(pose=p.as_pose(), weight=p.w))
        self.particle_pub.publish(msg)

    def scan_received(self, msg):
        self.last_scan_timestamp = msg.header.stamp
        # we throw away scans until we are done processing the previous scan
        # self.scan_to_process is set to None in the run_loop 
        if self.scan_to_process is None:
            self.scan_to_process = msg

def main(args=None):
    rclpy.init()
    n = ParticleFilter()
    rclpy.spin(n)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
