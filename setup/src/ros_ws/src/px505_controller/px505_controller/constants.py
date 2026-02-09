from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

### collection of constants used across the px505_controller package
# Setpoints
TAKE_OFF_ALTITUDE = 1  # meters

# Action related constants
class CONTROLLER_ACTION:
    HOVER    = "hover"
    TAKE_OFF = "take_off"
    LAND     = "land"
    TRACK    = "track"
    EXIT     = "exit"
    STAND_BY = "stand_by"

# Topic related constants
QOSP = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT
)

# physics
g = 9.8