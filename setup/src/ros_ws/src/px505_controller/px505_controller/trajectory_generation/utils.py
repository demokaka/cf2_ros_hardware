import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Quadrotor translation model (as in your screenshot)
#
# x_ddot = (T/m) * (cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))
# y_ddot = (T/m) * (cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))
# z_ddot = (T/m) * (cos(phi)*cos(theta)) - g
#
# Flat outputs: [x, y, z]^T, yaw psi chosen separately
#
# Inversion:
#   a_d = [xdd, ydd, zdd+g]
#   T   = m * ||a_d||
#   phi = asin((xdd*sin(psi) - ydd*cos(psi)) / ||a_d||)
#   theta = atan2(xdd*cos(psi) + ydd*sin(psi), zdd + g)
# ============================================================

# -------------------------
# Quintic trajectory utils
# -------------------------
def quintic_coeffs(p0, v0, a0, pT, vT, aT, T):
    T = float(T)
    M = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
        [0, 0, 2, 6*T, 12*T**2, 20*T**3],
    ], dtype=float)
    b = np.array([p0, v0, a0, pT, vT, aT], dtype=float)
    return np.linalg.solve(M, b)

def eval_quintic(c, t):
    c0, c1, c2, c3, c4, c5 = c
    p   = c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5
    dp  = c1 + 2*c2*t + 3*c3*t**2 + 4*c4*t**3 + 5*c5*t**4
    ddp = 2*c2 + 6*c3*t + 12*c4*t**2 + 20*c5*t**3
    return p, dp, ddp

def flat_poly_segment_3d(p0, v0, a0, pT, vT, aT, T, dt):
    p0 = np.asarray(p0, float); v0 = np.asarray(v0, float); a0 = np.asarray(a0, float)
    pT = np.asarray(pT, float); vT = np.asarray(vT, float); aT = np.asarray(aT, float)

    cx = quintic_coeffs(p0[0], v0[0], a0[0], pT[0], vT[0], aT[0], T)
    cy = quintic_coeffs(p0[1], v0[1], a0[1], pT[1], vT[1], aT[1], T)
    cz = quintic_coeffs(p0[2], v0[2], a0[2], pT[2], vT[2], aT[2], T)

    ts = np.arange(0.0, T + 1e-12, dt)

    p = np.zeros((3, ts.size))
    v = np.zeros((3, ts.size))
    a = np.zeros((3, ts.size))

    for k, t in enumerate(ts):
        p[0,k], v[0,k], a[0,k] = eval_quintic(cx, t)
        p[1,k], v[1,k], a[1,k] = eval_quintic(cy, t)
        p[2,k], v[2,k], a[2,k] = eval_quintic(cz, t)

    return ts, p, v, a

def flat_poly_path_3d(points, T_per_segment, dt, v_points=None, a_points=None):
    points = np.asarray(points, float)
    K = points.shape[0]
    if points.ndim != 2 or points.shape[1] != 3 or K < 2:
        raise ValueError("points must be (K,3) with K>=2")

    if np.isscalar(T_per_segment):
        Ts = np.full(K-1, float(T_per_segment))
    else:
        Ts = np.asarray(T_per_segment, float)
        if Ts.shape != (K-1,):
            raise ValueError("T_per_segment must be scalar or shape (K-1,)")

    if v_points is None:
        v_points = np.zeros((K,3), float)
    else:
        v_points = np.asarray(v_points, float)
        if v_points.shape != (K,3):
            raise ValueError("v_points must be (K,3)")

    if a_points is None:
        a_points = np.zeros((K,3), float)
    else:
        a_points = np.asarray(a_points, float)
        if a_points.shape != (K,3):
            raise ValueError("a_points must be (K,3)")

    ts_all = []
    p_all, v_all, a_all = [], [], []
    t_offset = 0.0

    for i in range(K-1):
        Tseg = Ts[i]
        ts, p, v, a = flat_poly_segment_3d(
            points[i], v_points[i], a_points[i],
            points[i+1], v_points[i+1], a_points[i+1],
            Tseg, dt
        )

        if i > 0:
            ts = ts[1:]
            p = p[:,1:]
            v = v[:,1:]
            a = a[:,1:]

        ts_all.append(t_offset + ts)
        p_all.append(p); v_all.append(v); a_all.append(a)
        t_offset += Tseg

    ts = np.concatenate(ts_all)
    p  = np.concatenate(p_all, axis=1)
    v  = np.concatenate(v_all, axis=1)
    a  = np.concatenate(a_all, axis=1)
    return ts, p, v, a

# -------------------------
# Flatness inversion
# -------------------------
def accel_to_thrust_angles(ax, ay, az, psi, m, g=9.81, eps=1e-9):
    ax = np.asarray(ax, float)
    ay = np.asarray(ay, float)
    az = np.asarray(az, float)
    psi = np.asarray(psi, float)

    adx = ax
    ady = ay
    adz = az + g

    norm_ad = np.sqrt(adx**2 + ady**2 + adz**2)
    norm_ad = np.maximum(norm_ad, eps)

    T = m * norm_ad

    sphi = (adx*np.sin(psi) - ady*np.cos(psi)) / norm_ad
    sphi = np.clip(sphi, -1.0, 1.0)
    phi = np.arcsin(sphi)

    theta = np.arctan2(adx*np.cos(psi) + ady*np.sin(psi), adz)

    return T, phi, theta

# -------------------------
# Forward accel + simulation
# -------------------------
def thrust_angles_to_accel(T, phi, theta, psi, m, g=9.81):
    ax = (T/m) * (np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi))
    ay = (T/m) * (np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi))
    az = (T/m) * (np.cos(phi)*np.cos(theta)) - g
    return ax, ay, az

def stack_segments(seg_list):
    """
    seg_list: list of (points, v_points, a_points)
    Stacks them, removing duplicate boundary point between segments.
    """
    P_all, V_all, A_all = [], [], []

    for k, (P, V, A) in enumerate(seg_list):
        if k > 0:
            P = P[1:]
            V = V[1:]
            A = A[1:]
        P_all.append(P); V_all.append(V); A_all.append(A)

    return np.vstack(P_all), np.vstack(V_all), np.vstack(A_all)

def make_takeoff_3pts(p0, z_hover=2.0, dz=0.0):
    """
    3-point vertical takeoff: start -> mid -> hover.
    - p0: current position (3,) [x,y,z]
    - z_hover: target altitude
    - dz: optional small offset to make mid-point not exactly halfway

    Returns: points, v_points, a_points
    """
    p0 = np.asarray(p0, float).reshape(3)
    x0, y0, z0 = p0

    z_mid = 0.5*(z0 + z_hover) + dz

    points = np.array([
        [x0, y0, z0],
        [x0, y0, z_mid],
        [x0, y0, z_hover],
    ], dtype=float)

    # Enforce vertical takeoff: zero horizontal velocity everywhere
    v_points = np.zeros_like(points)
    a_points = np.zeros_like(points)

    # If you specifically want final velocity 0 at hover, keep v_points[-1]=0 (already).
    return points, v_points, a_points


def make_landing_3pts(p_hover, z_land=0.0, dz=0.0):
    """
    3-point vertical landing: hover -> mid -> land.
    - p_hover: current position (3,) [x,y,z] (should be at hover altitude)
    - z_land: landing altitude (often 0)
    - dz: optional small offset for mid altitude

    Returns: points, v_points, a_points
    """
    p_hover = np.asarray(p_hover, float).reshape(3)
    x0, y0, z0 = p_hover

    z_mid = 0.5*(z0 + z_land) + dz

    points = np.array([
        [x0, y0, z0],
        [x0, y0, z_mid],
        [x0, y0, z_land],
    ], dtype=float)

    v_points = np.zeros_like(points)  # final velocity 0 at touchdown
    a_points = np.zeros_like(points)
    return points, v_points, a_points


def make_circle(
    p_center,
    R=1.0,
    z=None,
    N=50,
    v_circle=0.4,          # tangential speed along circle (m/s)
    v_depart=0.0,          # speed at center at start of depart (often 0)
    v_arrive=0.0,          # speed at center at end of return (often 0)
    v_trans=None,          # speed at circle entry/exit during transition (default: v_circle)
    n_trans=2,             # number of waypoints for depart and return (>=2, 3 is nice)
    start_angle=0.0,       # where the circle starts
):
    """
    Build waypoints for: center -> (smooth depart) -> circle -> (smooth return) -> center

    Returns:
      points  (K,3)
      v_points(K,3)
      a_points(K,3)  (zeros)
    """
    p_center = np.asarray(p_center, float).reshape(3)
    cx, cy, cz = p_center
    if z is None:
        z = cz

    if v_trans is None:
        v_trans = v_circle

    if n_trans < 2:
        raise ValueError("n_trans must be >= 2")

    # --- circle points (NOT including center) ---
    angles = start_angle + np.linspace(0.0, 2*np.pi, N, endpoint=False)
    circle_pts = np.column_stack([
        cx + R*np.cos(angles),
        cy + R*np.sin(angles),
        np.full(N, z),
    ])

    # Choose first and last circle point
    p_first = circle_pts[0]
    p_last  = circle_pts[-1]

    # Tangential velocity direction at an angle ang: [-sin(ang), cos(ang)]
    def v_tan_dir(ang):
        return np.array([-np.sin(ang), np.cos(ang), 0.0])

    v_first = v_trans * v_tan_dir(angles[0])
    v_last  = v_trans * v_tan_dir(angles[-1])

    # Center velocities for start/end (radial is often zero; keep configurable)
    # We'll set center departure velocity as radial toward first point (optional)
    # If v_depart is 0, you start from rest.
    radial_dir_first = (p_first - np.array([cx, cy, z]))
    radial_dir_first[2] = 0.0
    norm_rf = np.linalg.norm(radial_dir_first[:2])
    if norm_rf > 1e-12:
        radial_dir_first = radial_dir_first / norm_rf
    else:
        radial_dir_first = np.array([1.0, 0.0, 0.0])

    v_center_depart = v_depart * radial_dir_first
    v_center_arrive = np.zeros(3)  # direction irrelevant if magnitude 0
    if v_arrive != 0.0:
        # arrive direction opposite of radial from last point back to center
        radial_dir_last = (np.array([cx, cy, z]) - p_last)
        radial_dir_last[2] = 0.0
        norm_rl = np.linalg.norm(radial_dir_last[:2])
        if norm_rl > 1e-12:
            radial_dir_last = radial_dir_last / norm_rl
        else:
            radial_dir_last = np.array([1.0, 0.0, 0.0]
            )
        v_center_arrive = v_arrive * radial_dir_last

    center = np.array([cx, cy, z], dtype=float)

    # --- build depart points: center -> ... -> first circle point
    # Use linear interpolation in waypoint space; quintic will smooth between them using velocities.
    depart_pts = np.vstack([
        center + (i/(n_trans-1)) * (p_first - center)
        for i in range(n_trans)
    ])

    # --- build return points: last circle point -> ... -> center
    return_pts = np.vstack([
        p_last + (i/(n_trans-1)) * (center - p_last)
        for i in range(n_trans)
    ])

    # --- assemble full waypoint list
    # depart includes center and first circle point, so when we append circle we skip circle[0]
    # return includes last circle point and center, so we skip return[0]
    points = np.vstack([
        depart_pts,
        circle_pts[1:],     # continue around circle from second point
        return_pts[1:],     # go back to center
    ])

    # --- velocities at waypoints
    v_points = np.zeros_like(points)
    a_points = np.zeros_like(points)

    # indices for key points
    idx_center_start = 0
    idx_first_circle = n_trans - 1
    idx_last_circle  = idx_first_circle + (N - 1)  # because we added circle_pts[1:] (N-1 points)
    idx_center_end   = points.shape[0] - 1

    # set boundary/junction velocities (this is what kills spikes)
    v_points[idx_center_start] = v_center_depart
    v_points[idx_first_circle] = v_first
    v_points[idx_last_circle]  = v_last
    v_points[idx_center_end]   = v_center_arrive

    # also set velocities on intermediate circle waypoints (optional but helps smoothness)
    # (exclude first because already set)
    for i in range(1, N):
        # circle waypoint i in circle_pts maps to points index:
        # idx = idx_first_circle + (i - 0) ??? careful:
        # points contains depart_pts (n_trans), then circle_pts[1:] (i=1..N-1)
        # so circle_pts[i] goes to points index: idx_first_circle + (i-0) ???:
        # idx_first_circle is depart end (circle_pts[0])
        # circle_pts[1] is next => idx_first_circle + 1
        if i == 0:
            continue
        if i < N:
            idx = idx_first_circle + i
            v_points[idx] = v_trans * v_tan_dir(angles[i])

    # velocities on the transition interior points can be left 0; quintic will still be smooth.
    # If you want, you can taper them, but it's not required for continuity at joins.

    return points, v_points, a_points