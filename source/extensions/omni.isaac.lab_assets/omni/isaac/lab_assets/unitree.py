"""Portable path to the G1 asset bundled with this checkout."""
from pathlib import Path
G1_USD_PATH = str(Path(__file__).resolve().parents[6] / 'assets/g1/g1_29dof_modified_new_91.usd')
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
