import rclpy
import numpy as np
import pytest
from pf import ParticleFilter, Particle

class DummyNode:
    particle_cloud = []

def test_normalized_sums_to_one():
    rclpy.init()
    node = DummyNode()
    node.particle_cloud = [
        Particle(w=1),
        Particle(w=2),
        Particle(w=0.11),
    ]
    ParticleFilter.normalize_particles(node)
    assert sum([particle.w for particle in node.particle_cloud]) == pytest.approx(1.0)