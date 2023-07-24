import torch
import torch.nn as nn
import predify

class PCoderN(predify.modules.PCoderN):
    def __init__(
        self, pmodule: nn.Module, has_feedback: bool, random_init: bool,
        track_forward_terms: bool=False):

        super().__init__(pmodule, has_feedback, random_init)
        self.track_forward_terms = track_forward_terms

    def forward(self, ff: torch.Tensor, fb: torch.Tensor,
        target: torch.Tensor, build_graph: bool=False,
        ffm: float=None, fbm: float=None, erm: float=None):
        r"""
        Updates PCoder for one timestep.

        Args:
            ff (torch.Tensor): Feedforward drive.
            fb (torch.Tensor): Feedback drive (`None` if `has_feedback` is `False`).
            target (torch.Tensor): Target representation to compare with the prediction.
            build_graph (boolean): Indicates whether the computation graph
                should be built (set it to `True` during the training phase)
            ffm (float): The value of :math:`\beta`.
            fbm (float): The value of :math:`\lambda`.
            erm (float): The value of :math:`\alpha`.
        
        Returns:
            Tuple: the output representation and prediction
        """

        if self.rep is None:
            if self.random_init:
                self.rep = torch.randn(ff.size(), device=ff.device)
            else:
                self.rep = ff
        else:
            error_scale = self.prd.numel()/self.C_sqrt

            # Track represntation terms if desired
            if self.track_forward_terms:
                batch_size = ff.shape[0]
                err = error_scale*self.grd
                self.ff_norm = torch.norm(ff.reshape(batch_size, -1), dim=1)
                if self.has_feedback:
                    self.fb_norm = torch.norm(fb.reshape(batch_size, -1), dim=1)
                else:
                    self.fb_norm = torch.zeros(batch_size)
                self.mem_norm = torch.norm(self.rep.reshape(batch_size, -1), dim=1)
                self.err_norm = torch.norm(err.reshape(batch_size, -1), dim=1)

            # Calculate layer representation
            if self.has_feedback:
                self.rep = ffm*ff + fbm*fb + (1-ffm-fbm)*self.rep - erm*error_scale*self.grd
            else:
                self.rep = ffm*ff + (1-ffm)*self.rep - erm*error_scale*self.grd

        if self.C_sqrt == -1:
            self.compute_C_sqrt(target)

        with torch.enable_grad():
            if not self.rep.requires_grad:
                self.rep.requires_grad = True 
        
            self.prd = self.pmodule(self.rep)
            self.prediction_error  = nn.functional.mse_loss(self.prd, target)
            self.grd = torch.autograd.grad(self.prediction_error, self.rep, retain_graph=True)[0]
            
            if not build_graph:
                self.prd = self.prd.detach()
                self.grd = self.grd.detach()
                self.rep = self.rep.detach()
                self.prediction_error = self.prediction_error.detach()

        return self.rep, self.prd

