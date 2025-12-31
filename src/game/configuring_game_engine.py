from dataclasses import dataclass, field


@dataclass(frozen=True)
class RenderConfig:
    """Configuration for Pygame window and ModernGL FBO/Viewport."""

    RENDER_WIDTH: int = 500
    RENDER_HEIGHT: int = 500

    IS_HEADLESS: bool = False
    IS_VIS_MODE: bool = True

    # Make the player's optical flow faster/slower
    PLAYER_FLOW_MULT: float = 3.0
    # Make the cars go forward
    # 1.0: Same speed as road/scenery
    # >1: Moving towards player
    # <1: Moving away from player
    CAR_FLOW_MULT: float = -0.5


@dataclass(frozen=True)
class PhysicsConfig:
    """Configuration for game world physics and dynamics."""

    LANE_WIDTH: float = 2.5
    PLAYER_COLLISION_THRESH: float = 1.0

    PLAYER_COUNTERSTEER_FACTOR: float = 3.0
    PLAYER_TURN_SPEED: float = 10.0
    PLAYER_TURN_SPEED_WORLD_SPEED_FACTOR: float = 0.05
    PLAYER_COUNTERSTEER_DAMPING: float = 0.99
    PLAYER_TURN_DAMPING: float = 0.999

    INITIAL_WORLD_SPEED: float = 20.0
    MAX_WORLD_SPEED: float = 1000.0
    WORLD_ACCELERATION: float = 1.01
    DESPAWN_DISTANCE: float = -40.0
    GAMEOVER_TIME: float = 4.0


@dataclass(frozen=True)
class SpawningConfig:
    """Configuration for obstacle and scenery generation."""

    OPEN_LANE_LENGTH: float = 20.0
    NUM_INITIAL_OBSTACLES: int = 20
    NUM_INITIAL_BUILDINGS: int = 30
    OBSTACLE_LANE_DRIFT: float = 0.4
    INITIAL_SPAWN_DISTANCE: float = 25.0
    OBSTACLE_SPAWN_RATE: float = 0.6
    SCENERY_SPAWN_RATE: float = 5.0


@dataclass(frozen=True)
class PlayerConfig:
    WIDTH: float = 0.17
    HEIGHT: float = 0.65
    LENGTH: float = 0.5


@dataclass(frozen=True)
class CarConfig:
    WIDTH: float = 1.1
    HEIGHT: float = 0.55
    LENGTH: float = 2.0


@dataclass(frozen=True)
class BuildingConfig:
    """Configuration for building dimensions."""

    WIDTH: float = 10.0
    LENGTH: float = 30.0
    HEIGHT: float = 10.0

    SETBACK: float = 15.0


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for the 3D camera."""

    TRANSLATION_SMOOTH_SPEED: float = 0.5
    DIRECTION_SMOOTH_SPEED: float = 0.5
    FOV: float = 45.0
    FOLLOW_HEIGHT: float = 1.8
    FOLLOW_DISTANCE: float = 4.8
    MIN_RENDER_DISTANCE: float = 0.1
    MAX_RENDER_DISTANCE: float = 1000.0

    TARGET_DISTANCE: float = 5.0
    TARGET_HEIGHT: float = 1.0


@dataclass(frozen=True)
class ColorConfig:
    """Standardized color values (R, G, B floats)."""

    BG: tuple[float, float, float] = (0.1, 0.1, 0.1)
    PLAYER: tuple[float, float, float] = (0.0, 1.0, 0.0)
    BUILDING: tuple[float, float, float] = (0.5, 0.5, 0.5)
    GROUND_BASE: tuple[float, float, float] = (0.2, 0.2, 0.2)
    GROUND_CHECKER_SCALE: float = 10.0
    CAR: tuple[float, float, float] = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class GameEngineConfig:
    """Primary configuration container for the entire game engine."""

    render: RenderConfig = field(default_factory=RenderConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    spawning: SpawningConfig = field(default_factory=SpawningConfig)
    player: PlayerConfig = field(default_factory=PlayerConfig)
    car: CarConfig = field(default_factory=CarConfig)
    building: BuildingConfig = field(default_factory=BuildingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
