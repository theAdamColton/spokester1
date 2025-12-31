import random
import pygame
import moderngl
import numpy as np
import math

from src.game.configuring_game_engine import GameEngineConfig

# --- Shaders ---

CTX_VERT_SHADER = """
#version 330
// Current Matrices
uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;

// Previous Matrices (for Flow)
uniform mat4 m_proj_prev;
uniform mat4 m_view_prev;
uniform mat4 m_model_prev;

uniform float u_is_checkerboard;
in vec3 in_position;

out vec3 v_world_pos;
out float v_is_checkerboard;
out vec4 v_curr_pos; // Clip space current
out vec4 v_prev_pos; // Clip space previous

void main() {
    vec4 world_pos = m_model * vec4(in_position, 1.0);
    
    // Current Frame Position
    vec4 clip_pos = m_proj * m_view * world_pos;
    gl_Position = clip_pos;
    
    // Previous Frame Position
    // We recalculate where this specific vertex was in the previous frame
    vec4 world_pos_prev = m_model_prev * vec4(in_position, 1.0);
    vec4 clip_pos_prev = m_proj_prev * m_view_prev * world_pos_prev;

    v_world_pos = world_pos.xyz;
    v_is_checkerboard = u_is_checkerboard;
    v_curr_pos = clip_pos;
    v_prev_pos = clip_pos_prev;
}
"""

CTX_FRAG_SHADER = """
#version 330
uniform vec3 u_color;
uniform float u_scroll_offset;
uniform float u_checker_scale;

in vec3 v_world_pos;
in float v_is_checkerboard;
in vec4 v_curr_pos;
in vec4 v_prev_pos;

// Output to multiple targets: 0 = RGB Color, 1 = Optical Flow (Float)
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_flow;

void main() {
    // --- 1. Color Calculation ---
    vec4 color_out;
    if (v_is_checkerboard > 0.5) {
        float scrolling_z = v_world_pos.z + u_scroll_offset;
        float check_scale = u_checker_scale;
        
        int x_int = int(floor(v_world_pos.x / check_scale));
        int z_int = int(floor(scrolling_z / check_scale));

        if (mod(x_int + z_int, 2) == 0) {
            color_out = vec4(0.2, 0.2, 0.2, 1.0);
        } else {
            color_out = vec4(0.3, 0.3, 0.3, 1.0);
        }
    } else {
        color_out = vec4(u_color, 1.0);
    }
    f_color = color_out;

    // --- 2. Optical Flow Calculation ---
    // Convert to Normalized Device Coordinates (NDC) -> [-1, 1]
    vec2 ndc_curr = v_curr_pos.xy / v_curr_pos.w;
    vec2 ndc_prev = v_prev_pos.xy / v_prev_pos.w;

    // Calculate delta in UV space [0, 1]
    // UV = (NDC + 1) * 0.5
    // Delta UV = (NDC_curr - NDC_prev) * 0.5
    vec2 flow_uv = (ndc_curr - ndc_prev) * 0.5;

    // Output: x=u, y=v, z=validity mask (1.0)
    f_flow = vec3(flow_uv, 1.0);
}
"""

# 2D Screen Quad Shader
SCREEN_VERT = """
#version 330
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = in_uv;
}
"""

SCREEN_FRAG = """
#version 330
uniform sampler2D u_texture;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(u_texture, v_uv);
}
"""


# --- Math Helpers ---
def create_perspective_matrix(fov_y, aspect, near, far):
    f = 1.0 / math.tan(fov_y / 2.0)
    matrix = np.zeros((4, 4), dtype="f4")
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = -1.0
    matrix[3, 2] = (2.0 * far * near) / (near - far)
    return matrix


def create_look_at(eye, target, up):
    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    matrix = np.identity(4, dtype="f4")
    matrix[0, :3] = x_axis
    matrix[1, :3] = y_axis
    matrix[2, :3] = z_axis
    matrix[:3, 3] = -np.dot(matrix[:3, :3], eye)
    return matrix.T


def create_transformation_matrix(x, y, z, sx, sy, sz):
    m = np.identity(4, dtype="f4")
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    m[3, 0] = x
    m[3, 1] = y
    m[3, 2] = z
    return m


def random_a_b(rng, a=0.0, b=1.0):
    u = rng.random()
    return u * (b - a) + a


class GameEngine:
    def __init__(self, config: GameEngineConfig = GameEngineConfig()):
        self.config = config
        cfg = self.config

        if cfg.render.IS_VIS_MODE:
            # Window width is 3x render width to show Seg, Depth, Flow side-by-side
            self.window_size = (
                cfg.render.RENDER_WIDTH * 3,
                cfg.render.RENDER_HEIGHT,
            )
        else:
            self.window_size = (cfg.render.RENDER_WIDTH, cfg.render.RENDER_HEIGHT)

        if cfg.render.IS_HEADLESS:
            pygame.display.init()
            pygame.init()
            pygame.display.set_caption("Spokester")
            self.window = None
            self.ctx = moderngl.create_standalone_context(backend="egl")

        else:
            pygame.init()
            pygame.display.set_caption("Spokester")
            self.window = pygame.display.set_mode(
                self.window_size,
                pygame.OPENGL | pygame.DOUBLEBUF,
            )
            self.ctx = moderngl.create_context()

        self._init_3d_assets()
        self._init_2d_screen_assets()

        # Matrices tracking for Optical Flow
        self.m_proj_prev = np.identity(4, dtype="f4")
        self.m_view_prev = np.identity(4, dtype="f4")
        self.is_first_frame = True

        # Physics tracking for reconstructing previous model matrices
        self.last_dt = 0.0
        self.last_player_dx = 0.0
        self.last_world_dz = 0.0

        self.reset_game()

    def reset_game(self, seed=None):
        cfg = self.config
        self.rng = random.Random(seed)

        self.world_speed = cfg.physics.INITIAL_WORLD_SPEED
        self.world_distance = 0

        self.lanes = [-cfg.physics.LANE_WIDTH, 0, cfg.physics.LANE_WIDTH]
        self.player_lane_idx = 1
        self.prev_player_lane_idx = self.player_lane_idx
        self.player_steer_offset = 0.0
        self.player_coords = [self.lanes[self.player_lane_idx], 0.0, 0.0]
        self.player_vel_x = 0.0

        self.gameover = False
        self.gameover_time = 0

        self.open_lane = self.rng.randint(0, len(self.lanes) - 1)
        self.open_lane_z = 0

        self.obstacles = []
        for _ in range(cfg.spawning.NUM_INITIAL_OBSTACLES):
            self._spawn_random_obstacle(
                (cfg.spawning.INITIAL_SPAWN_DISTANCE, cfg.camera.MAX_RENDER_DISTANCE)
            )

        self.scenery = []
        for _ in range(cfg.spawning.NUM_INITIAL_BUILDINGS):
            self._spawn_random_scenery(
                (cfg.physics.DESPAWN_DISTANCE, cfg.camera.MAX_RENDER_DISTANCE)
            )

        self.score = 0
        self.is_first_frame = True

        # Reset physics deltas
        self.last_player_dx = 0.0
        self.last_world_dz = 0.0

        return self.render_and_get_observation()

    def _spawn_random_obstacle(self, distance_range: tuple[float, float] | None = None):
        cfg = self.config

        spawnable_lanes = list(range(len(self.lanes)))
        spawnable_lanes.pop(self.open_lane)

        lane_idx = self.rng.choice(spawnable_lanes)
        x = self.lanes[lane_idx]
        drift = random_a_b(
            self.rng,
            -cfg.physics.LANE_WIDTH * cfg.spawning.OBSTACLE_LANE_DRIFT,
            cfg.physics.LANE_WIDTH * cfg.spawning.OBSTACLE_LANE_DRIFT,
        )
        x = x + drift
        distance = cfg.camera.MAX_RENDER_DISTANCE
        if distance_range is not None:
            distance = random_a_b(self.rng, *distance_range)
        self.obstacles.append({"type": "car", "x": x, "z": distance})

    def _spawn_random_scenery(self, distance_range: tuple[float, float] | None = None):
        cfg = self.config
        x_spawn_distance = (
            cfg.physics.LANE_WIDTH + cfg.building.WIDTH + cfg.building.SETBACK
        )
        x = self.rng.choice([-x_spawn_distance, x_spawn_distance])
        distance = cfg.camera.MAX_RENDER_DISTANCE
        if distance_range is not None:
            distance = random_a_b(self.rng, *distance_range)
        self.scenery.append({"type": "building", "x": x, "z": distance})

    def _init_3d_assets(self):
        """Setup FBO, Shaders, and VBOs for the 3D world."""
        cfg = self.config

        # FBO Setup
        # Attachment 0: Color (Segmentation)
        self.fbo_color = self.ctx.texture(
            (cfg.render.RENDER_WIDTH, cfg.render.RENDER_HEIGHT), 3
        )
        # Attachment 1: Optical Flow (Float32 for precision)
        # Storing (u, v, valid)
        self.fbo_flow = self.ctx.texture(
            (cfg.render.RENDER_WIDTH, cfg.render.RENDER_HEIGHT), 3, dtype="f4"
        )

        self.fbo_depth = self.ctx.depth_texture(
            (cfg.render.RENDER_WIDTH, cfg.render.RENDER_HEIGHT)
        )

        # Create FBO with multiple render targets
        self.fbo = self.ctx.framebuffer([self.fbo_color, self.fbo_flow], self.fbo_depth)

        # Geometry (Cube)
        coords_2d = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype="f4")
        coords_yz_neg_x = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype="f4")
        ones_neg = np.full((4, 1), -1.0, dtype="f4")
        ones_pos = np.full((4, 1), 1.0, dtype="f4")

        vertices = (
            np.concatenate(
                [
                    np.hstack((coords_2d, ones_neg)),
                    np.hstack((coords_2d, ones_pos)),
                    np.hstack((ones_neg, coords_yz_neg_x)),
                    np.hstack((ones_pos, coords_yz_neg_x)),
                    np.hstack((coords_2d[:, :1], ones_neg, coords_2d[:, 1:])),
                    np.hstack((coords_2d[:, :1], ones_pos, coords_2d[:, 1:])),
                ]
            )
            .flatten()
            .astype("f4")
        )

        face_indices_pattern = np.array([0, 1, 2, 2, 3, 0], dtype="i4")
        indices = np.concatenate(
            [face_indices_pattern + 4 * i for i in range(6)]
        ).astype("i4")

        self.prog_3d = self.ctx.program(
            vertex_shader=CTX_VERT_SHADER, fragment_shader=CTX_FRAG_SHADER
        )
        self.prog_3d["u_is_checkerboard"].value = 0
        self.prog_3d["u_scroll_offset"].value = 0.0
        self.prog_3d["u_checker_scale"].value = cfg.color.GROUND_CHECKER_SCALE

        self.vbo_cube = self.ctx.buffer(vertices)
        self.ibo_cube = self.ctx.buffer(indices)
        self.vao_cube = self.ctx.vertex_array(
            self.prog_3d, [(self.vbo_cube, "3f", "in_position")], self.ibo_cube
        )

    def _init_2d_screen_assets(self):
        cfg = self.config
        # Quad for displaying texture (Full screen -1 to 1)
        quad_data = np.array(
            [
                # x, y, u, v
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype="f4",
        )
        self.prog_screen = self.ctx.program(
            vertex_shader=SCREEN_VERT, fragment_shader=SCREEN_FRAG
        )
        self.vbo_quad = self.ctx.buffer(quad_data)
        self.vao_quad = self.ctx.vertex_array(
            self.prog_screen, [(self.vbo_quad, "2f 2f", "in_vert", "in_uv")]
        )

        self.display_texture = self.ctx.texture(self.window_size, 3)
        self.display_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _render_box(
        self,
        x,
        y,
        z,
        sx,
        sy,
        sz,
        color,
        is_checkerboard=False,
        scroll_offset=0.0,
        prev_x=None,
        prev_y=None,
        prev_z=None,
    ):
        """
        Renders a box and calculates its current and previous model matrices.
        If prev_coords are None, assumes the object didn't move relative to the world origin.
        """
        # Current Model Matrix
        m_model = create_transformation_matrix(x, y + sy, z, sx, sy, sz)

        # Previous Model Matrix
        px = x if prev_x is None else prev_x
        py = y if prev_y is None else prev_y
        pz = z if prev_z is None else prev_z
        m_model_prev = create_transformation_matrix(px, py + sy, pz, sx, sy, sz)

        self.prog_3d["m_model"].write(m_model.tobytes())
        self.prog_3d["m_model_prev"].write(m_model_prev.tobytes())

        self.prog_3d["u_color"].value = color
        self.prog_3d["u_is_checkerboard"].value = 1 if is_checkerboard else 0
        self.prog_3d["u_scroll_offset"].value = scroll_offset
        self.vao_cube.render()

    def _lerp_value(self, value, attr: str, speed: float = 0.1):
        if not hasattr(self, attr):
            setattr(self, attr, value)
            return value
        speed = min(speed, 1.0)
        speed = max(speed, 0.0)
        smooth_value = getattr(self, attr)
        smooth_value = smooth_value + (value - smooth_value) * speed
        setattr(self, attr, smooth_value)
        return smooth_value

    def _render_scene_to_fbo(self, dt: float = 1 / 30):
        """Render 3D world to FBO."""
        cfg = self.config

        self.fbo.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(*cfg.color.BG)  # Clears color attachment 0
        # Manual clear for flow attachment (0,0,0)
        self.fbo.clear(
            0.0,
            0.0,
            0.0,
            1.0,
            viewport=(0, 0, cfg.render.RENDER_WIDTH, cfg.render.RENDER_HEIGHT),
        )

        # --- Camera Calculation ---
        target_camera_coords = np.array([self.player_coords[0], 0, 0], dtype="f4")
        ideal_eye_offset = np.array(
            [0, cfg.camera.FOLLOW_HEIGHT, -cfg.camera.FOLLOW_DISTANCE], dtype="f4"
        )
        ideal_eye = target_camera_coords + ideal_eye_offset
        ideal_target_offset = np.array(
            [0, cfg.camera.TARGET_HEIGHT, cfg.camera.TARGET_DISTANCE], dtype="f4"
        )
        ideal_target = target_camera_coords + ideal_target_offset

        self._lerp_value(
            ideal_eye, "camera_eye", cfg.camera.TRANSLATION_SMOOTH_SPEED * dt
        )
        self._lerp_value(
            ideal_target, "camera_target", cfg.camera.DIRECTION_SMOOTH_SPEED * dt
        )

        up = np.array([0.0, 1.0, 0.0], dtype="f4")
        m_view = create_look_at(self.camera_eye, self.camera_target, up)

        m_proj = create_perspective_matrix(
            math.radians(cfg.camera.FOV),
            cfg.render.RENDER_WIDTH / cfg.render.RENDER_HEIGHT,
            cfg.camera.MIN_RENDER_DISTANCE,
            cfg.camera.MAX_RENDER_DISTANCE,
        )

        # Handle Previous Matrices for Flow
        if self.is_first_frame:
            self.m_view_prev = m_view.copy()
            self.m_proj_prev = m_proj.copy()

        # Upload Matrices
        self.prog_3d["m_view"].write(m_view.tobytes())
        self.prog_3d["m_proj"].write(m_proj.tobytes())
        self.prog_3d["m_view_prev"].write(self.m_view_prev.tobytes())
        self.prog_3d["m_proj_prev"].write(self.m_proj_prev.tobytes())

        # --- Render Objects ---

        # 1. Player
        # Player moves in X based on last_player_dx.
        # Player moves in Y/Z: Z is fixed at 0.0 relative to world root

        px, py, pz = self.player_coords
        self._render_box(
            px,
            py,
            pz,
            cfg.player.WIDTH,
            cfg.player.HEIGHT,
            cfg.player.LENGTH,
            cfg.color.PLAYER,
            # Revert x position
            prev_x=px - self.last_player_dx * cfg.render.PLAYER_FLOW_MULT,
            prev_y=py,
            prev_z=pz,
        )

        # 2. Ground
        # Ground object is static
        # Manually calculate a prev_z
        # Current box z is 1.0
        # Prev box z is 1 + self.last_world_dz
        self._render_box(
            0,
            -0.1,
            1.0,
            100.0,
            0.1,
            cfg.camera.MAX_RENDER_DISTANCE,
            cfg.color.GROUND_BASE,
            is_checkerboard=True,
            scroll_offset=self.world_distance,
            prev_z=1 + self.last_world_dz,
        )

        # 3. Obstacles (Cars)
        # In step(), obs["z"] decreases by world_speed * dt.
        # So prev_z = curr_z + world_speed * dt (which is last_world_dz)

        for obs in self.obstacles:
            if obs["type"] == "car":
                prev_z = obs["z"] + self.last_world_dz * cfg.render.CAR_FLOW_MULT
                # Body
                self._render_box(
                    obs["x"],
                    0,
                    obs["z"],
                    cfg.car.WIDTH,
                    cfg.car.HEIGHT,
                    cfg.car.LENGTH,
                    cfg.color.CAR,
                    prev_x=obs["x"],
                    prev_z=prev_z,
                )
                # Roof
                self._render_box(
                    obs["x"],
                    1.0,
                    obs["z"],
                    0.7,
                    0.4,
                    1.0,
                    cfg.color.CAR,
                    prev_x=obs["x"],
                    prev_z=prev_z,
                )

        # 4. Scenery (Buildings)
        for scene in self.scenery:
            if scene["type"] == "building":
                prev_z = scene["z"] + self.last_world_dz
                self._render_box(
                    scene["x"],
                    0,
                    scene["z"],
                    cfg.building.WIDTH,
                    cfg.building.HEIGHT,
                    cfg.building.LENGTH,
                    cfg.color.BUILDING,
                    is_checkerboard=True,
                    scroll_offset=self.world_distance,
                    prev_x=scene["x"],
                    prev_z=prev_z,
                )

        # Save matrices for next frame
        self.m_view_prev = m_view.copy()
        self.m_proj_prev = m_proj.copy()
        self.is_first_frame = False

    def render_and_get_observation(self, dt: float = 1 / 30):
        cfg = self.config

        self._render_scene_to_fbo(dt)

        # 1. Read Segmentation (Color) - Attachment 0
        seg_data = self.fbo.read(components=3, attachment=0)
        seg_np = np.frombuffer(seg_data, dtype="u1").reshape(
            (cfg.render.RENDER_HEIGHT, cfg.render.RENDER_WIDTH, 3)
        )

        # 2. Read Optical Flow - Attachment 1 (Float32)
        flow_data = self.fbo.read(components=3, attachment=1, dtype="f4")
        flow_np = np.frombuffer(flow_data, dtype="f4").reshape(
            (cfg.render.RENDER_HEIGHT, cfg.render.RENDER_WIDTH, 3)
        )

        # 3. Read Depth - Attachment Depth
        depth_data = self.fbo.read(components=1, attachment=-1, dtype="f4")
        depth_np = np.frombuffer(depth_data, dtype="f4").reshape(
            (cfg.render.RENDER_HEIGHT, cfg.render.RENDER_WIDTH)
        )

        # Flip and Orient
        seg_np = seg_np[::-1, ::-1]
        flow_np = flow_np[::-1, ::-1]

        # Invert flow, vu -> uv
        flow_np = flow_np * -1

        depth_np = depth_np[::-1, ::-1]
        depth_np = 1 - depth_np

        return depth_np, seg_np, flow_np

    def step(self, action: int, dt: float = 1 / 30):
        cfg = self.config

        # Track Physics Deltas for Flow reconstruction
        prev_player_x = self.player_coords[0]

        self.world_speed = self.world_speed + cfg.physics.WORLD_ACCELERATION * dt
        self.world_speed = min(self.world_speed, cfg.physics.MAX_WORLD_SPEED)

        self.prev_player_lane_idx = self.player_lane_idx
        if action == 1 and self.player_lane_idx > 0:
            self.player_lane_idx -= 1
        elif action == 2 and self.player_lane_idx < 2:
            self.player_lane_idx += 1

        step_distance_traveled = self.world_speed * dt
        self.world_distance += step_distance_traveled

        self.open_lane_z += step_distance_traveled
        if self.open_lane_z > cfg.spawning.OPEN_LANE_LENGTH:
            self.open_lane_z = 0
            self.open_lane = self.rng.randint(0, len(self.lanes) - 1)

        # Store Z displacement for obstacles
        self.last_world_dz = step_distance_traveled

        for obs in self.obstacles:
            obs["z"] -= self.world_speed * dt
        for scene in self.scenery:
            scene["z"] -= self.world_speed * dt

        if self.rng.random() < cfg.spawning.OBSTACLE_SPAWN_RATE * dt:
            self._spawn_random_obstacle()
        if self.rng.random() < cfg.spawning.SCENERY_SPAWN_RATE * dt:
            self._spawn_random_scenery()

        self.obstacles = [
            o for o in self.obstacles if o["z"] > cfg.physics.DESPAWN_DISTANCE
        ]
        self.scenery = [
            o for o in self.scenery if o["z"] > cfg.physics.DESPAWN_DISTANCE
        ]

        for obs in self.obstacles:
            if obs["type"] == "car":
                if (abs(obs["z"] - 0) < 2.5) and (
                    abs(obs["x"] - self.player_coords[0])
                    < cfg.physics.PLAYER_COLLISION_THRESH
                ):
                    self.gameover = True

        reward = step_distance_traveled
        if self.gameover:
            self.gameover_time += dt
            reward = 0
            self.player_coords[2] -= step_distance_traveled

        if not self.gameover:
            target_x = self.lanes[self.player_lane_idx]
            current_x = self.player_coords[0]
            player_changed_lane = self.player_lane_idx != self.prev_player_lane_idx

            world_speed_mult = (
                1 + self.world_speed * cfg.physics.PLAYER_TURN_SPEED_WORLD_SPEED_FACTOR
            )

            if player_changed_lane:
                direction = 1 if target_x > current_x else -1
                vel_towards_target = self.player_vel_x * direction
                correction_factor = 1.0

                if vel_towards_target > 0:
                    cutoff_speed = 1.0 * world_speed_mult
                    ratio = vel_towards_target / cutoff_speed
                    correction_factor = max(0.0, 1.0 - ratio)

                self.player_steer_offset = (
                    -direction
                    * cfg.physics.PLAYER_COUNTERSTEER_FACTOR
                    * world_speed_mult
                    * correction_factor
                )

            self.player_steer_offset *= (
                1 - cfg.physics.PLAYER_COUNTERSTEER_DAMPING
            ) ** (dt * world_speed_mult)

            effective_target = target_x + self.player_steer_offset
            player_turn_speed = cfg.physics.PLAYER_TURN_SPEED * world_speed_mult

            # Spring acceleration
            acceleration = (effective_target - current_x) * player_turn_speed

            self.player_vel_x += acceleration * dt

            damping_factor = (1.0 - cfg.physics.PLAYER_TURN_DAMPING) ** dt
            self.player_vel_x *= damping_factor

            self.player_coords[0] += self.player_vel_x * dt

        # Calculate how much player moved in X
        self.last_player_dx = self.player_coords[0] - prev_player_x

        self.score += reward
        done = self.gameover_time >= cfg.physics.GAMEOVER_TIME

        self.last_dt = dt

        # Return all 3 buffers
        obs = self.render_and_get_observation(dt)
        return obs, reward, done

    def _render_frame(self, frame: np.ndarray):
        # Flip up down (OpenGL texture coordinate system)
        frame = frame[::-1]

        assert (self.window_size[1], self.window_size[0], 3) == frame.shape
        assert frame.dtype == np.uint8

        frame = np.ascontiguousarray(frame)
        self.display_texture.write(frame.tobytes())

        self.ctx.screen.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.display_texture.use(location=0)
        self.vao_quad.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()

    def display_to_screen(self, *args):
        cfg = self.config
        if cfg.render.IS_VIS_MODE:
            return self._display_seg_depth_flow(*args)

        frame, *_ = args

        return self._render_frame(frame)

    def _display_seg_depth_flow(self, depth_np, seg_np, flow_np):
        """Visualize buffers to the PyGame window."""
        cfg = self.config
        if cfg.render.IS_HEADLESS:
            raise ValueError("is headless")

        # Process Flow for Visualization
        scale = 20.0
        shift = 0.5
        flow_np = (flow_np * scale + shift) * 255
        flow_np = np.clip(flow_np, 0, 255).astype("u1")
        flow_np[..., 2] = 0

        # Process Depth for Visualization
        depth_min = 0.0001
        depth_max = 0.02
        depth_gamma = 0.5
        depth_np = (depth_np - depth_min) / (depth_max - depth_min)
        depth_np = np.sign(depth_np) * np.abs(depth_np) ** depth_gamma
        depth_np = depth_np * 255
        depth_np = np.clip(depth_np, 0, 255).astype("u1")
        depth_np = np.stack((depth_np,) * 3, axis=-1)

        # Combine side-by-side (Seg | Depth | Flow)
        # Note: Input depth_np is already formatted as RGB uint8
        combined = np.hstack((seg_np, depth_np, flow_np))

        return self._render_frame(combined)
