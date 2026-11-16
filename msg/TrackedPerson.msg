# One person tracked in the map frame (people_tracker_node: constant-velocity Kalman filter per person).
uint32 id
float64 x            # map frame, m
float64 y
float64 vx           # map frame, m/s
float64 vy
float32 radius       # m, from the smoothed detection width
float32 pos_std      # 1-sigma position uncertainty, m
float32 vel_std      # 1-sigma velocity uncertainty, m/s
uint32 hits          # detections that updated this track
time last_seen       # stamp of the last detection that updated it
