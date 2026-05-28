import copy

import torch


class ModelEMA:
    """Model Exponential Moving Average (GPU-resident，老路径，保留兼容)。"""

    def __init__(self, model, decay=0.9999, device=None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CPUEMA:
    """CPU 储存的 EMA：避免在 GPU 上 deepcopy 整份 Whisper-large + Qwen 引发 OOM。

    - 初始化：从 model.state_dict() 直接克隆到 CPU（不 deepcopy 整个模块）。
    - update：每步把 model.state_dict() 的浮点张量 .to('cpu') 后做 EMA 更新；非浮点张量
      （如 token type ids 等 buffers）直接保留最新值。
    - apply_to / restore：在评估前把 EMA 权重灌回模型，评估后还原；这样不需要在 GPU
      上常驻一份 EMA 副本。
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, p in model.state_dict().items():
                self.shadow[name] = p.detach().to("cpu", copy=True)
        self._backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, p in model.state_dict().items():
            shadow_p = self.shadow.get(name)
            if shadow_p is None:
                self.shadow[name] = p.detach().to("cpu", copy=True)
                continue
            if not p.is_floating_point():
                shadow_p.copy_(p.detach().to("cpu"))
                continue
            shadow_p.mul_(self.decay).add_(
                p.detach().to("cpu", dtype=shadow_p.dtype), alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        """把 EMA 权重灌回 model（用于评估）。backup **必须存在 CPU**，否则
        Whisper-large + Qwen 这种大模型在 GPU 上复制一份会直接 OOM /
        触发系统内存 swap → eval 速度暴跌 30 倍。
        """
        self._backup = {
            name: p.detach().to("cpu", copy=True) for name, p in model.state_dict().items()
        }
        sd = model.state_dict()
        for name, shadow_p in self.shadow.items():
            if name in sd:
                # .copy_() 自动跨设备拷贝（CPU shadow → GPU model）
                sd[name].copy_(shadow_p.to(dtype=sd[name].dtype))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """评估后把训练权重恢复。CPU backup → GPU model 通过 copy_ 自动跨设备。"""
        if self._backup is None:
            return
        sd = model.state_dict()
        for name, val in self._backup.items():
            if name in sd:
                sd[name].copy_(val)
        self._backup = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().to("cpu", copy=True) for k, v in state.items()}
