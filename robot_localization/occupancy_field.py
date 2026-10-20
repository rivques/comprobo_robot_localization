""" An implementation of an occupancy field that calculates the distance
to the closest obstacle for any point on the map. Modified from stock OccupancyField
to also support returning values some distance outside of the map. """

import rclpy
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
import numpy as np
from sklearn.neighbors import NearestNeighbors

class OccupancyField(object):
    """ Stores an occupancy field for an input map.  An occupancy field returns
        the distance to the closest obstacle for any coordinate in the map
        Attributes:
            map: the map to localize against (nav_msgs/OccupancyGrid)
            closest_occ: the distance for each entry in the OccupancyGrid to
            the closest obstacle
    """

    def __init__(self, node, extra_meters=0):
        """
        Initialize the OccupancyField. Arguments:
        node: ParticleFilter instance.
        extra_meters: How many meters outside of the provided
            map to calculate distances for. Useful for when LiDar
            rays originating in the map land outside of the map.
        """
        # grab the map from the map server
        self.cli = node.create_client(GetMap, 'map_server/map')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            node.get_logger().info('service not available, waiting again...')
        self.future = self.cli.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(node, self.future)
        self.map: OccupancyGrid = self.future.result().map
        node.get_logger().info("map received width: {0} height: {1}, resolution: {2}".format(self.map.info.width, self.map.info.height, self.map.info.resolution))
        
        # expand the map in all directions so we can answer questions about out-of-bounds lidar points
        old_width = self.map.info.width
        old_height = self.map.info.height
        self.actual_width_m = old_width * self.map.info.resolution
        self.actual_height_m = old_height * self.map.info.resolution
        self.actual_origin_x_m = self.map.info.origin.position.x
        self.actual_origin_y_m = self.map.info.origin.position.y
        cells_to_add = int(extra_meters / self.map.info.resolution)
        print(f"Adding {cells_to_add} cells in each direction")
        self.map.info.width += 2 * cells_to_add
        self.map.info.height += 2 * cells_to_add
        self.map.info.origin.position.x -= cells_to_add * self.map.info.resolution
        self.map.info.origin.position.y -= cells_to_add * self.map.info.resolution

        # The coordinates of each grid cell in the map
        X = np.zeros((self.map.info.width*self.map.info.height, 2))

        # while we're at it let's count the number of occupied cells
        total_occupied = 0
        curr = 0
        for i in range(self.map.info.width):
            for j in range(self.map.info.height):
                # occupancy grids are stored in row major order
                # compensate for the extra cells so we don't index out of bounds
                old_i = i - cells_to_add
                old_j = j - cells_to_add
                if old_i >= 0 and old_i < old_width and old_j >= 0 and old_j < old_height:
                    # this point would have landed on the original map
                    ind = old_i + old_j * old_width
                    # check if there's something here
                    if self.map.data[ind] > 0:
                        total_occupied += 1
                X[curr, 0] = float(i)
                X[curr, 1] = float(j)
                curr += 1

        # The coordinates of each occupied grid cell in the map
        occupied = np.zeros((total_occupied, 2))
        curr = 0
        for i in range(self.map.info.width):
            for j in range(self.map.info.height):
                # occupancy grids are stored in row major order
                # compensate for the extra cells so we don't index out of bounds
                old_i = i - cells_to_add
                old_j = j - cells_to_add
                if old_i >= 0 and old_i < old_width and old_j >= 0 and old_j < old_height:
                    # this point would have landed on the original map
                    ind = old_i + old_j * old_width
                    if self.map.data[ind] > 0:
                        # there's something here, note that
                        occupied[curr, 0] = float(i)
                        occupied[curr, 1] = float(j)
                        curr += 1
        node.get_logger().info("building ball tree")
        # use super fast scikit learn nearest neighbor algorithm
        nbrs = NearestNeighbors(n_neighbors=1,
                                algorithm="ball_tree").fit(occupied)
        node.get_logger().info("finding neighbors")
        distances, indices = nbrs.kneighbors(X)

        node.get_logger().info("populating occupancy field")
        self.closest_occ = np.zeros((self.map.info.width, self.map.info.height))
        curr = 0
        for i in range(self.map.info.width):
            for j in range(self.map.info.height):
                self.closest_occ[i, j] = \
                    distances[curr][0]*self.map.info.resolution
                curr += 1
        self.occupied = occupied
        node.get_logger().info("occupancy field ready")

    def get_obstacle_bounding_box(self):
        """
        Returns: the upper and lower bounds of x and y such that the resultant
        bounding box contains all of the obstacles in the map.  The format of
        the return value is ((x_lower, x_upper), (y_lower, y_upper))
        """
        lower_bounds = self.occupied.min(axis=0)
        upper_bounds = self.occupied.max(axis=0)
        r = self.map.info.resolution
        return ((lower_bounds[0]*r + self.map.info.origin.position.x,
                 upper_bounds[0]*r + self.map.info.origin.position.x),
                (lower_bounds[1]*r + self.map.info.origin.position.y,
                 upper_bounds[1]*r + self.map.info.origin.position.y))

    def get_closest_obstacle_distance(self, x, y):
        """ Compute the closest obstacle to the specified (x,y) coordinate in
            the map. If the x,y coordinate is out of bounds, no big deal."""
        x_coord = (x - self.map.info.origin.position.x)/self.map.info.resolution
        y_coord = (y - self.map.info.origin.position.y)/self.map.info.resolution
        if type(x) is np.ndarray:
            x_coord = x_coord.astype(np.int)
            y_coord = y_coord.astype(np.int)
        else:
            x_coord = int(x_coord)
            y_coord = int(y_coord)

        is_valid = (x_coord >= 0) & (y_coord >= 0) & (x_coord < self.map.info.width) & (y_coord < self.map.info.height)
        if type(x) is np.ndarray:
            distances = np.float('nan')*np.ones(x_coord.shape)
            distances[is_valid] = self.closest_occ[x_coord[is_valid], y_coord[is_valid]]
            return distances
        else:
            return self.closest_occ[x_coord, y_coord] if is_valid else float('nan')