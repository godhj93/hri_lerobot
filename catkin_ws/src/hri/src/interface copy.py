import mujoco
import numpy as np
from termcolor import colored

class SimulatedRobot:
    def __init__(self, m, d) -> None:
        """
        :param m: mujoco model
        :param d: mujoco data
        """
        self.m = m
        self.d = d

        # PID gains for 6D error (3D position + 3D orientation)
        self.Kp = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.Ki = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) * 0.0
        self.Kd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) * 10.0

        # PID error storage
        self.integral_error = np.zeros(6)
        self.prev_error = np.zeros(6)

    def _apply_pid(self, error, dt):
        """
        Apply PID control to the given error signal.
        :param error: 6D error (position + orientation)
        :param dt: timestep from the model
        :return: control signal after PID
        """
        # Proportional
        p_term = self.Kp * error

        # Integral
        self.integral_error += error * dt
        i_term = self.Ki * self.integral_error

        # Derivative
        d_term = self.Kd * (error - self.prev_error) / dt
        self.prev_error = error

        # PID output
        control_signal = p_term + i_term + d_term
        return control_signal

    def _pos2pwm(self, pos: np.ndarray) -> np.ndarray:
        return (pos / 3.14 + 1.) * 4096

    def _pwm2pos(self, pwm: np.ndarray) -> np.ndarray:
        return (pwm / 2048 - 1) * 3.14 

    def _pwm2norm(self, x: np.ndarray) -> np.ndarray:
        return x / 4096

    def _norm2pwm(self, x: np.ndarray) -> np.ndarray:
        return x * 4096

    def read_position(self) -> np.ndarray:
        return self.d.qpos[:6]

    def read_velocity(self):
        return self.d.qvel

    def read_ee_pos(self, joint_name='end_effector'):
        joint_id = self.m.body(joint_name).id
        return self.d.geom_xpos[joint_id]

    def set_target_pos(self, target_pos):
        self.d.ctrl = target_pos

    def inverse_kinematics(self, ee_target_pos, rate=0.2, joint_name='end_effector'):
        joint_id = self.m.body(joint_name).id
        ee_pos = self.d.geom_xpos[joint_id]
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jac, None, joint_id)
        qdot = np.dot(np.linalg.pinv(jac[:, :6]), ee_target_pos - ee_pos)
        qpos = self.read_position()
        q_target_pos = qpos + qdot * rate
        return q_target_pos

    def inverse_kinematics_rot(self, ee_target_pos, rate=0.2, joint_name='end_effector'):
        joint_id = self.m.body(joint_name).id
        ee_pos = self.d.geom_xpos[joint_id]
        ee_rot = self.d.geom_xmat[joint_id].reshape(3, 3)

        z_axis = np.array([0, 0, 1], dtype=np.float64)
        x_axis = np.array([1, 0, 0], dtype=np.float64)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        ee_target_rot = np.column_stack((x_axis, y_axis, z_axis))

        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBody(self.m, self.d, jacp, jacr, joint_id)

        error_pos = ee_target_pos - ee_pos

        current_quat = np.zeros(4)
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(current_quat, ee_rot.flatten())
        mujoco.mju_mat2Quat(target_quat, ee_target_rot.flatten())

        error_quat = np.zeros(4)
        mujoco.mju_subQuat(error_quat, target_quat, current_quat)

        error_rot = np.zeros(3)
        mujoco.mju_quat2Vel(error_rot, error_quat, 1.0)

        error = np.hstack((error_pos, error_rot))
        jac = np.vstack((jacp, jacr))

        lambda_identity = 1e-4 * np.eye(6)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lambda_identity, error)

        q = self.d.qpos.copy()
        mujoco.mj_integratePos(self.m, q, dq * rate)

        np.clip(q[:6], self.m.jnt_range[:6, 0], self.m.jnt_range[:6, 1], out=q[:6])
        self.d.ctrl[:6] = q[:6]
        mujoco.mj_step(self.m, self.d)

    def inverse_kinematics_rot_backup_6DOF(self, ee_target_pos, ee_target_rot, rate=0.2, joint_name='end_effector'):
        """
        PID를 적용한 예시:
        error를 기반으로 PID를 계산한 후,
        PID 출력값(control_signal)을 이용해 dq를 계산하도록 수정.
        """
        joint_id = self.m.body(joint_name).id

        ee_pos = self.d.geom_xpos[joint_id]
        ee_rot = self.d.geom_xmat[joint_id]
        error = np.zeros(6)
        error_pos = error[:3]
        error_rot = error[3:]
        site_quat = np.zeros(4)
        site_target_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)

        diag = 1e-4 * np.identity(6)
        integration_dt = 1.0 / 2

        # compute the jacobian
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jacp, jacr, joint_id)
        jac = np.vstack([jacp, jacr])

        # Orientation error.
        mujoco.mju_mat2Quat(site_quat, ee_rot)
        mujoco.mju_mat2Quat(site_target_quat, ee_target_rot)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, site_target_quat, site_quat_conj)
        mujoco.mju_quat2Vel(error_rot, error_quat, 1.0)

        error_pos = ee_target_pos - ee_pos
        error = np.hstack([error_pos, error_rot])

        # 여기서 PID 적용
        dt = self.m.opt.timestep  # 시뮬레이션 timestep
        control_signal = self._apply_pid(error, dt)

        # PID 출력인 control_signal을 사용하여 dq 계산
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, control_signal)

        q = self.d.qpos.copy()
        mujoco.mj_integratePos(self.m, q, dq, integration_dt)

        np.clip(q[:6], *self.m.jnt_range.T[:, :6], out=q[:6])
        self.d.ctrl[:6] = q[:6]

        mujoco.mj_step(self.m, self.d)
        
        return self.d.ctrl[:6]

    def inverse_kinematics_rot_backup_5DOF(self, ee_target_pos, ee_target_rot, rate=0.2, joint_name='end_effector'):
        joint_id = self.m.body(joint_name).id
        ee_pos = self.d.geom_xpos[joint_id]
        ee_rot = self.d.geom_xmat[joint_id]
        error = np.zeros(6)
        error_pos = error[:3]
        error_rot = error[3:]
        site_quat = np.zeros(4)
        site_target_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)

        diag = 1e-4 * np.identity(5)
        integration_dt = 1.0

        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jacp, jacr, joint_id)
        
        jac = np.vstack([jacp, jacr])

        mujoco.mju_mat2Quat(site_quat, ee_rot)
        mujoco.mju_mat2Quat(site_target_quat, ee_target_rot)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, site_target_quat, site_quat_conj)
        mujoco.mju_quat2Vel(error_rot, error_quat, 1.0)

        error_pos = ee_target_pos - ee_pos
        error = np.hstack([error_pos, error_rot])

        dq = jac[:5, :5].T @ np.linalg.solve(jac[:5, :5] @ jac[:5, :5].T + diag, error[:5])

        q = self.d.qpos.copy()
        mujoco.mj_integratePos(self.m, q, dq, integration_dt)

        np.clip(q[:5], *self.m.jnt_range.T[:, :5], out=q[:5])
        self.d.ctrl[:5] = q[:5]

        print(colored(f"Target joint position: {np.round(q[:5], 2)}", 'red'))
        mujoco.mj_step(self.m, self.d)
        
        return self.d.ctrl[:5]

    def inverse_kinematics_rot_backup_4DOF(self, ee_target_pos, ee_target_rot, rate=0.2, joint_name='end_effector'):
        joint_id = self.m.body(joint_name).id
        ee_pos = self.d.geom_xpos[joint_id]
        ee_rot = self.d.geom_xmat[joint_id]
        error = np.zeros(6)
        error_pos = error[:3]
        error_rot = error[3:]
        site_quat = np.zeros(4)
        site_target_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)

        diag = 1e-4 * np.identity(4)
        integration_dt = 1.0 / 2

        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jacp, jacr, joint_id)
        
        jac = np.vstack([jacp, jacr])

        mujoco.mju_mat2Quat(site_quat, ee_rot)
        mujoco.mju_mat2Quat(site_target_quat, ee_target_rot)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, site_target_quat, site_quat_conj)
        mujoco.mju_quat2Vel(error_rot, error_quat, 1.0)

        error_pos = ee_target_pos - ee_pos
        error = np.hstack([error_pos, error_rot])
        
        # 여기서 PID 적용
        dt = self.m.opt.timestep  # 시뮬레이션 timestep
        control_signal = self._apply_pid(error, dt)
        
        dq = jac[:4, :4].T @ np.linalg.solve(jac[:4, :4] @ jac[:4, :4].T + diag, control_signal[:4])

        q = self.d.qpos.copy()[:4]
        print(q.shape, dq.shape, self.m)
        mujoco.mj_integratePos(self.m, q, dq, integration_dt)

        np.clip(q[:4], *self.m.jnt_range.T[:, :4], out=q[:4])
        self.d.ctrl[:4] = q[:4]

        print(colored(f"Target joint position: {np.round(q[:4], 2)}", 'red'))
        mujoco.mj_step(self.m, self.d)
        
        return self.d.ctrl[:4]
