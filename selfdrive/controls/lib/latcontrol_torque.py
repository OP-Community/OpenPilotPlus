import math
import numpy as np

from cereal import log
from opendbc.car.interfaces import LatControlInputs
from opendbc.car.vehicle_model import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.pid import PIDController
from opendbc.car.toyota.values import CAR

# At higher speeds (25+mph) we can assume:
# Lateral acceleration achieved by a specific car correlates to
# torque applied to the steering rack. It does not correlate to
# wheel slip, or to speed.

# This controller applies torque to achieve desired lateral
# accelerations. To compensate for the low speed effects we
# use a LOW_SPEED_FACTOR in the error. Additionally, there is
# friction in the steering wheel that needs to be overcome to
# move it at all, this is compensated for too.

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [15, 13, 10, 5]

# Pre-compute low speed factors for faster lookup
LOW_SPEED_FACTORS = np.array([y**2 for y in LOW_SPEED_Y])


def fast_low_speed_factor(v_ego):
    """Fast low speed factor calculation without np.interp"""
    if v_ego <= LOW_SPEED_X[0]:
        return LOW_SPEED_FACTORS[0]
    elif v_ego >= LOW_SPEED_X[-1]:
        return LOW_SPEED_FACTORS[-1]
    else:
        # Find the appropriate segment
        for i in range(len(LOW_SPEED_X) - 1):
            if LOW_SPEED_X[i] <= v_ego <= LOW_SPEED_X[i + 1]:
                # Linear interpolation
                x0, x1 = LOW_SPEED_X[i], LOW_SPEED_X[i + 1]
                y0, y1 = LOW_SPEED_FACTORS[i], LOW_SPEED_FACTORS[i + 1]
                return y0 + (y1 - y0) * (v_ego - x0) / (x1 - x0)
    return LOW_SPEED_FACTORS[0]  # fallback


def fast_curvature_interpolation(v_ego, curvature_vm, curvature_pose):
    """Fast curvature interpolation between 2.0 and 5.0 m/s"""
    if v_ego <= 2.0:
        return curvature_vm
    elif v_ego >= 5.0:
        return curvature_pose
    else:
        # Linear interpolation between 2.0 and 5.0 m/s
        t = (v_ego - 2.0) / 3.0
        return curvature_vm + t * (curvature_pose - curvature_vm)


class LatControlTorque(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.pid = PIDController(self.torque_params.kp, self.torque_params.ki,
                             k_f=self.torque_params.kf, pos_limit=self.steer_max, neg_limit=-self.steer_max)
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.use_steering_angle = self.torque_params.useSteeringAngle
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.CP = CP  # Store CP for car identification

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def update(self, active, CS, VM, params, steer_limited_by_controls, desired_curvature, calibrated_pose, curvature_limited):
    pid_log = log.ControlsState.LateralTorqueState.new_message()

    if not active:
      output_torque = 0.0
      pid_log.active = False
    else:
      # Calculate curvature from vehicle model (optimized)
      actual_curvature_vm = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
      roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY

      if self.use_steering_angle and self.CP.carFingerprint != CAR.TOYOTA_PRIUS:
        actual_curvature = actual_curvature_vm
        curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
      else:
        assert calibrated_pose is not None
        actual_curvature_pose = calibrated_pose.angular_velocity.yaw / CS.vEgo
        actual_curvature = fast_curvature_interpolation(CS.vEgo, actual_curvature_vm, actual_curvature_pose)
        curvature_deadzone = 0.0

      # Optimize lateral acceleration calculations - compute v_ego_squared once
      v_ego_squared = CS.vEgo * CS.vEgo
      desired_lateral_accel = desired_curvature * v_ego_squared
      actual_lateral_accel = actual_curvature * v_ego_squared
      lateral_accel_deadzone = curvature_deadzone * v_ego_squared

      # Fast low speed factor calculation
      low_speed_factor = fast_low_speed_factor(CS.vEgo)

      # Optimize setpoint and measurement calculations
      setpoint = desired_lateral_accel + low_speed_factor * desired_curvature
      measurement = actual_lateral_accel + low_speed_factor * actual_curvature
      gravity_adjusted_lateral_accel = desired_lateral_accel - roll_compensation

      # Create LatControlInputs objects once and reuse
      inputs_setpoint = LatControlInputs(setpoint, roll_compensation, CS.vEgo, CS.aEgo)
      inputs_measurement = LatControlInputs(measurement, roll_compensation, CS.vEgo, CS.aEgo)
      inputs_ff = LatControlInputs(gravity_adjusted_lateral_accel, roll_compensation, CS.vEgo, CS.aEgo)

      # Optimize torque calculations by reducing function call overhead
      torque_from_setpoint = self.torque_from_lateral_accel(inputs_setpoint, self.torque_params,
                                                            setpoint, lateral_accel_deadzone, friction_compensation=False, gravity_adjusted=False)
      torque_from_measurement = self.torque_from_lateral_accel(inputs_measurement, self.torque_params,
                                                               measurement, lateral_accel_deadzone, friction_compensation=False, gravity_adjusted=False)
      pid_log.error = float(torque_from_setpoint - torque_from_measurement)

      ff = self.torque_from_lateral_accel(inputs_ff, self.torque_params,
                                          desired_lateral_accel - actual_lateral_accel, lateral_accel_deadzone, friction_compensation=True,
                                          gravity_adjusted=True)

      freeze_integrator = steer_limited_by_controls or CS.steeringPressed or CS.vEgo < 5
      output_torque = self.pid.update(pid_log.error,
                                      feedforward=ff,
                                      speed=CS.vEgo,
                                      freeze_integrator=freeze_integrator)

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(self.pid.d)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(-output_torque)
      pid_log.actualLateralAccel = float(actual_lateral_accel)
      pid_log.desiredLateralAccel = float(desired_lateral_accel)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_controls, curvature_limited))

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log