import rclpy
import numpy as np
import pytest
from pf import ParticleFilter, Particle
from helper_functions import TFHelper

# surely there's a better way to do this
class InertTFHelper:
    def convert_pose_to_xy_and_theta(self, pose):
        return TFHelper.convert_pose_to_xy_and_theta(None, pose)
    def angle_normalize(self, z):
        return TFHelper.angle_normalize(None, z)
class InertOccupancyField:
    actual_origin_y_m = 0
    actual_origin_x_m = 0
    actual_height_m = 5
    actual_width_m = 5

class InertNode:
    # workaround since an actual node would hang on calling mapserver etc on init
    odom_pose = Particle(x=0,y=0,theta=0).as_pose()
    current_odom_xy_theta = (0,0,0)
    particle_cloud = []
    transform_helper = InertTFHelper()
    occupancy_field = InertOccupancyField()

def test_normalized_sums_to_one():
    rclpy.init()
    node = InertNode()
    node.particle_cloud = [
        Particle(w=1),
        Particle(w=2),
        Particle(w=0.11),
    ]
    ParticleFilter.normalize_particles(node)
    assert sum([particle.w for particle in node.particle_cloud]) == pytest.approx(1.0)

def test_update_particles_with_odom():
    node = InertNode()
    node.particle_cloud = [
            Particle(x=0, y=0, theta=0),
            Particle(x=0, y=1, theta=-np.pi/2),
    ]
    # move nowhere
    new_odom_pose = Particle(x=0, y=0, theta=0).as_pose() # using this as a shortcut for xytheta tuple to Pose
    node.odom_pose = new_odom_pose
    ParticleFilter.update_particles_with_odom(node, ODOM_NOISE=0)
    # nothing should have moved
    assert [(p.x, p.y, p.theta) for p in node.particle_cloud] == pytest.approx([(0, 0, 0), (0, 1, -np.pi/2)])

    # move 1 meter forward
    new_odom_pose = Particle(x=1, y=0, theta=0).as_pose()
    node.odom_pose = new_odom_pose
    ParticleFilter.update_particles_with_odom(node, ODOM_NOISE=0)
    # particles should have moved accordingly
    assert [(float(p.x), float(p.y), float(p.theta)) for p in node.particle_cloud] == [pytest.approx((1, 0, 0)), pytest.approx((0, 0, -np.pi/2))]

    # move 1 meter sideways and be rotated by 45deg
    # remember we started at 1,0,0
    new_odom_pose = Particle(x=1, y=1, theta=np.pi/4).as_pose()
    node.odom_pose = new_odom_pose
    ParticleFilter.update_particles_with_odom(node, ODOM_NOISE=0)
    # particles should have moved accordingly
    assert [(p.x, p.y, p.theta) for p in node.particle_cloud] == [pytest.approx((1, 1, np.pi/4)), pytest.approx((1, 0, -np.pi/4))]

if __name__ == "__main__":
    test_update_particles_with_odom()