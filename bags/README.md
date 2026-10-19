# bags
This directory contains several rosbag recordings related to our project. 
* `macfirst_floor_take_1`: A raw recording of a Turtlebot driving around the first floor of the Miller Academic Center, starting in the 126/128 alcove area.
* `macfirst_floor_take_2`: Same Turtlebot, same building, but a much shorter recording and a more ambiguous starting point in the hallway.
* `pf_localizing_mac_take_1`: Our final particle filter responding to the playback of `macfirst_floor_take_1`, broadcasting its particle cloud on `/particle_cloud` and its map -> odom transform on `tf`.
* `pf_localizing_mac_take_2`: Our final particle filter responding to the playback of `macfirst_floor_take_2`.
The recordings of our filter's response are after final tuning and are quality, fast convergences. Sometimes our luck is worse and the filter takes longer to converge, but it usually makes it there eventually.