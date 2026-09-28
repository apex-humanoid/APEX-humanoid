# ===== Multi-Teacher (TorchScript-friendly) =====
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import List, Optional, Union

from rsl_rl.utils import resolve_nn_activation



class ObsAdapter(nn.Module):
    """
    Select / transform a unified teacher observation into the input each teacher expects.
    Indices are relative to unified_teacher_obs (not student obs).
    """
    def __init__(self, select_indices: List[int]):
        super().__init__()
        self.register_buffer("idxs", torch.as_tensor(select_indices, dtype=torch.long))
    def forward(self, unified_teacher_obs: torch.Tensor) -> torch.Tensor:
        return unified_teacher_obs.index_select(-1, self.idxs)


class CompositeStudentTeacher(nn.Module):
    """
    One trainable student; K frozen teachers (eager or TorchScript).
    Distillation calls:
      - act(obs)                -> student rollout actions (sample)
      - act_inference(obs)      -> student mean actions (for BC gradient)
      - evaluate(teacher_obs)   -> per-env teacher mean actions (BC targets)
    teacher_obs layout: [teacher_id | unified_teacher_obs...]
    """
    is_recurrent = False

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,        # kept for signature compatibility with runner
        num_actions: int,
        student_hidden_dims: List[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 0.1,
        teachers: Optional[List[nn.Module]] = None,   # can be eager or ScriptModule
        adapters_indices: list[list[int]] | None = None,
        **kwargs,
    ):
        if kwargs:
            print("CompositeStudentTeacher.__init__ got unexpected args (ignored):", list(kwargs.keys()))
        super().__init__()

        act_fn = resolve_nn_activation(activation)

        # ---- Student MLP (simple, fast) ----
        layers: List[nn.Module] = []
        layers += [nn.Linear(num_student_obs, student_hidden_dims[0]), act_fn]
        for i in range(len(student_hidden_dims)):
            is_last = (i == len(student_hidden_dims) - 1)
            in_f   = student_hidden_dims[i]
            out_f  = num_actions if is_last else student_hidden_dims[i+1]
            layers.append(nn.Linear(in_f, out_f))
            if not is_last:
                layers.append(act_fn)
        self.student = nn.Sequential(*layers)

        # action noise (for rollout sampling API parity)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Optional[Normal] = None
        self.update_distribution(torch.zeros(1, num_student_obs))
        Normal.set_default_validate_args = False

        # ---- Teachers & adapters ----
        self.teachers = nn.ModuleList(teachers or [])


        D_u = num_teacher_obs 

        if adapters_indices is not None:
            parsed_lists = []
            for spec in adapters_indices:
                idx_list = _parse_index_spec(spec, D_u)
                parsed_lists.append(idx_list)
            built_adapters = [ObsAdapter(idx_list) for idx_list in parsed_lists]
            self.adapters = nn.ModuleList(built_adapters)

        # freeze teachers if any are eager modules
        for t in self.teachers:
            try:
                t.eval()
                for p in t.parameters():
                    p.requires_grad_(False)
            except Exception:
                # TorchScript modules don't expose .parameters() the same way; safe to ignore
                pass

        # lets runner know teachers exist
        self.loaded_teacher = len(self.teachers) > 0

    # -------- student side ----------
    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.student(observations)
        std  = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        # self.update_distribution(observations)
        # return self.distribution.sample()
        return self.student(observations)


    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        # student mean (pre-tanh if you use squashing downstream)
        return self.student(observations)
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def walk_joint_idx(self) -> List[int]:
        """Return the indices of the walk teacher's joints in the unified observation."""
        
        return [0,1,3,4,6,7,9,10,13,14, 17,18]

    # -------- teacher side ----------


    @torch.no_grad()
    def evaluate(self, teacher_observations: torch.Tensor) -> torch.Tensor:
        """
        teacher_observations: [B, 1 + D_u]
          - col 0: teacher_id (float or int; we cast to long)
          - cols 1..: unified_teacher_obs (D_u)
        Returns: [B, num_actions] per-env teacher mean actions as BC targets.
        """
        if len(self.teachers) == 0:
            raise RuntimeError("No teachers attached to CompositeStudentTeacher.")
        tid  = teacher_observations[:, 0].long()   # [B]
        uobs = teacher_observations[:, 1:]         # [B, D_u]
        B    = uobs.shape[0]
        act_dim = self.student[-1].out_features
        out  = uobs.new_zeros(B, act_dim)

        for k, (teacher, normalizer,adapter) in enumerate(zip(self.teachers, self.normalizers, self.adapters)):
            mask = (tid == k)
            # print('mask:', mask)
            if not mask.any():
                continue
            # print('teacher obs:', teacher_observations.shape)
            # print('teacher obs masked:', teacher_observations[mask].shape)
            xk = adapter(teacher_observations[mask])               # adapt unified obs -> this teacher's input
            mu = teacher(normalizer(xk))   # eager or TorchScript
            if k == 0:
                envs = mask.nonzero(as_tuple=True)[0]
                out[envs.unsqueeze(1), self.walk_joint_idx] = mu
            else:
                out[mask, :] = mu

        return out

    # ---- loader for TorchScript teachers ----
    def load_scripted_teachers_from_paths(
        self,
        paths: List[str],
        map_location: Optional[Union[str, torch.device]] = None,
        print_summary: bool = True,
    ) -> None:
        """
        Load K TorchScript teachers from K .pt files (saved with torch.jit.save).
        Each ScriptModule must accept a tensor of shape [B, obs_dim_k] and return a tensor [B, num_actions] (mean actions).
        """
        if len(paths) != len(self.adapters):
            raise ValueError(f"Need one path per adapter/teacher: got {len(paths)} paths for {len(self.adapters)} adapters")
        self.teachers = nn.ModuleList([
            torch.jit.load(p, map_location=map_location).actor for p in paths
        ])
        self.normalizers = nn.ModuleList([
            torch.jit.load(p, map_location=map_location).normalizer for p in paths
        ])
        self.normalizers.eval()

        # torchscript modules are already eval/frozen
        self.loaded_teacher = True
        if print_summary:
            for i, p in enumerate(paths):
                print(f"[CompositeStudentTeacher] Loaded TorchScript teacher #{i} from: {p}")

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """

        # check if state_dict contains teacher and student or just teacher parameters
        print('composite student-teacher load_state_dict keys:', state_dict.keys())
        filtered = {
        k: v for k, v in state_dict.items()
        if not (k.startswith("teachers.") or k.startswith("normalizers."))
    }
        super().load_state_dict(filtered, strict=strict)
        
        return True
    # RNN no-ops
    def reset(self, dones=None, hidden_states=None): pass
    def get_hidden_states(self): return None
    def detach_hidden_states(self, dones=None): pass


def _parse_index_spec(spec, D_u: int) -> list[int]:
    """
    Parse an index spec into a list[int], where D_u is unified_teacher_obs dim.
    Accepts:
      - list[int]
      - str like "a:b", "a:", ":b", "i,j,k"
      - ["a:b"] (single-item list of str)
    """
    # unwrap single-item list of str
    if isinstance(spec, list) and len(spec) == 1 and isinstance(spec[0], str):
        spec = spec[0]

    # already a list[int]
    if isinstance(spec, list):
        # ensure ints
        return [int(i) for i in spec]

    if not isinstance(spec, str):
        raise ValueError(f"Unsupported adapters_indices entry: {spec!r} (type {type(spec)})")

    parts = [p.strip() for p in spec.split(",") if p.strip() != ""]
    out: list[int] = []
    for p in parts:
        if ":" in p:
            a_str, b_str = p.split(":", 1)
            a = int(a_str) if a_str != "" else 0
            b = int(b_str) if b_str != "" else D_u
            if not (0 <= a <= b <= D_u):
                raise ValueError(f"Range '{p}' is out of bounds for D_u={D_u}")
            out.extend(range(a, b))
        else:
            # single integer token
            i = int(p)
            if not (0 <= i < D_u):
                raise ValueError(f"Index '{i}' out of bounds for D_u={D_u}")
            out.append(i)
    return out