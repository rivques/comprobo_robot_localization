from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pytest
from pf import ParticleFilter, Particle
from helper_functions import TFHelper
from geometry_msgs.msg import Pose

# this is ugly, but allows us to test the particle filter without needing to start up ROS at all
class InertTFHelper:
    def convert_pose_to_xy_and_theta(self, pose):
        return TFHelper.convert_pose_to_xy_and_theta(None, pose)
    def angle_normalize(self, z):
        return TFHelper.angle_normalize(None, z)
    def fix_map_to_odom_transform(self, robot_pose, odom_pose):
        pass
    
class InertOccupancyField:
    actual_origin_y_m = 0
    actual_origin_x_m = 0
    actual_height_m = 5
    actual_width_m = 5
    
    def get_closest_obstacle_distance(self, x, y):
        # simplified hardcoded map:
        # there's a wall enclosing the entire map
        
        # this assumes the origin is zero for simplicity
        wall_distance = min(abs(x), abs(y), abs(x - self.actual_width_m), abs(y - self.actual_height_m))
        return wall_distance

class InertParam:
    value: Any

    PARAMS = {
        "n_particles": 10_000, # for testing only
        "proportion_uniform": 0.3,
        "normalization_base": 0.4,
        "odom_noise": 0 # for testing only
    }

    def __init__(self, param_name) -> None:
        try:
            self.value = self.PARAMS[param_name]
        except KeyError as e:
            raise NameError(f"Tried to access a ROS param that does not exist: {e}")

class InertNode:
    # workaround since an actual node would hang on calling mapserver etc on init
    odom_pose = Particle(x=0,y=0,theta=0).as_pose()
    robot_pose: Pose = None
    current_odom_xy_theta = (0,0,0)
    particle_cloud : List[Particle] = []
    transform_helper = InertTFHelper()
    occupancy_field = InertOccupancyField()

    def get_parameter(self, param_name):
        return InertParam(param_name)
    def normalize_particles(self):
        return ParticleFilter.normalize_particles(self)
    def update_robot_pose(self):
        return ParticleFilter.update_robot_pose(self)

def test_normalize_particles():
    node = InertNode()
    node.particle_cloud = [
        Particle(w=1),
        Particle(w=2),
        Particle(w=0.11),
    ]
    ParticleFilter.normalize_particles(node)
    assert sum([particle.w for particle in node.particle_cloud]) == pytest.approx(1.0)
    # TODO: add'l normalized testing

def test_resample_particles():
    node = InertNode()
    initial_particles = [
        Particle(x=0, y=0, theta=0, w=0.1),
        Particle(x=3, y=1, theta=np.pi, w=0.7),
        Particle(x=0, y=4, theta=0, w=0.2)
    ]
    node.particle_cloud = initial_particles
    ParticleFilter.resample_particles(node)

    counts = [0, 0, 0, 0]
    for particle in node.particle_cloud:
        if particle.equal_pose(initial_particles[0]):
            counts[0] += 1
        elif particle.equal_pose(initial_particles[1]):
            counts[1] += 1
        elif particle.equal_pose(initial_particles[2]):
            counts[2] += 1
        else:
            counts[3] += 1

    proportions = [count / node.get_parameter("n_particles").value for count in counts]
    expected_proportions = [particle.w * (1 - node.get_parameter("proportion_uniform").value) for particle in initial_particles]
    expected_proportions.append(node.get_parameter("proportion_uniform").value)
    
    assert proportions == pytest.approx(expected_proportions, abs=0.01)

def test_update_robot_pose():
    node = InertNode()
    node.particle_cloud = [
        Particle(x=0, y=0, theta=0, w=1),
        Particle(x=3, y=1, theta=np.pi, w=2),
        Particle(x=0, y=4, theta=0, w=0.11),
    ]
    ParticleFilter.update_robot_pose(node)
    updated_pose = node.transform_helper.convert_pose_to_xy_and_theta(node.robot_pose)
    assert updated_pose[0] == 0 and updated_pose[1] == 4 and updated_pose[2] == 0

def test_update_particles_with_odom():
    node = InertNode()
    node.particle_cloud = [
        Particle(x=0, y=0, theta=0),
        Particle(x=0, y=1, theta=-np.pi/2),
    ]
    # move nowhere
    new_odom_pose = Particle(x=0, y=0, theta=0).as_pose() # using this as a shortcut for xytheta tuple to Pose
    node.odom_pose = new_odom_pose
    ParticleFilter.update_particles_with_odom(node)
    # nothing should have moved
    assert [(p.x, p.y, p.theta) for p in node.particle_cloud] == pytest.approx([(0, 0, 0), (0, 1, -np.pi/2)])

    # move 1 meter forward
    new_odom_pose = Particle(x=1, y=0, theta=0).as_pose()
    node.odom_pose = new_odom_pose
    ParticleFilter.update_particles_with_odom(node)
    # particles should have moved accordingly
    assert [(float(p.x), float(p.y), float(p.theta)) for p in node.particle_cloud] == [pytest.approx((1, 0, 0)), pytest.approx((0, 0, -np.pi/2))]

    # move 1 meter sideways and be rotated by 45deg
    # remember we started at 1,0,0
    new_odom_pose = Particle(x=1, y=1, theta=np.pi/4).as_pose()
    node.odom_pose = new_odom_pose
    ParticleFilter.update_particles_with_odom(node)
    # particles should have moved accordingly
    assert [(p.x, p.y, p.theta) for p in node.particle_cloud] == [pytest.approx((1, 1, np.pi/4)), pytest.approx((1, 0, -np.pi/4))]

def test_update_particles_with_laser():
    node = InertNode()
    node.particle_cloud = [
        Particle(x=3, y=3, theta=np.pi/2), # perfect
        Particle(x=2, y=2, theta=-np.pi/2), # mirror, we shouldn't be able to tell this apart
        Particle(x=1, y=1, theta=np.pi), # wrong on all counts
        Particle(x=3, y=3, theta=-np.pi/2) # right position, wrong orientation
    ]
    # construct our test laser scans:
    # we'll pretend we're actually at (3, 3, np.pi/2)
    thetas = np.linspace(-np.pi, np.pi, 360)
    r = np.zeros_like(thetas)
    for i, theta in enumerate(thetas):
        if theta > -np.pi/4 and theta < np.pi/4:
            # we see the top wall
            # cos(theta) = 2m / r
            r[i] = 2 / np.cos(theta)
        elif theta > -3*np.pi/4 and theta <= -np.pi/4:
            # we see the right wall
            # cos(theta + pi/2) = 2m / r
            r[i] = 2 / np.cos(theta + np.pi/2)
        else:
            # we see something faaaaaaar away
            r[i] = np.inf

    ParticleFilter.update_particles_with_laser(node, r, thetas)
    assert node.particle_cloud[0].w == pytest.approx(0)
    assert node.particle_cloud[1].w == pytest.approx(0)
    assert node.particle_cloud[2].w > 0.2
    assert node.particle_cloud[3].w > 0.2

def test_initialize_particle_cloud():
    node = InertNode()
    ParticleFilter.initialize_particle_cloud(node)

    particle_positions = np.array([[p.x, p.y, p.theta] for p in node.particle_cloud])

    assert len(node.particle_cloud) == node.get_parameter("n_particles").value # we should have as many particles as we ask for
    assert len(np.unique(particle_positions.flatten())) == len(particle_positions.flatten()) # no two particles should be in the same place
    
    # all particles should be on the map
    assert np.all(particle_positions[:, 0] >= node.occupancy_field.actual_origin_x_m)
    assert np.all(particle_positions[:, 1] >= node.occupancy_field.actual_origin_y_m)
    assert np.all(particle_positions[:, 2] >= -np.pi)
    assert np.all(particle_positions[:, 0] <= node.occupancy_field.actual_origin_x_m + node.occupancy_field.actual_width_m)
    assert np.all(particle_positions[:, 1] <= node.occupancy_field.actual_origin_y_m + node.occupancy_field.actual_height_m)
    assert np.all(particle_positions[:, 2] <= np.pi)