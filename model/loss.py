import torch.nn as nn

class KpLoss(nn.Module):
    def __init__(self, position_loss_weight=.7):
        super().__init__()
        self.position_loss_weight = position_loss_weight
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """Custom loss function for Key points detection. We calculate 
        position loss (Mean Squared Error loss) for the position of the point
        and visibility loss (Cross Entropy Loss) if the point is visible or not.

        Pred and target are tensors of shape [batch_size, #_keypoints, 3]. First 2 
        elements are point position and the last is point visibility.

        The final loss is calculated by the following formula:
        position_loss * self.position_loss_weight + visibility_loss * (1 - self.position_loss_weight)

        Args:
            pred (_type_): predicted values
            target (_type_): labels

        Returns:
            _type_: tuple of calculated loss and dict of separated 
            position and visibility losses.
        """
        # Both should have shape [batch_size, num_kps, 3]
        assert pred.shape == target.shape

        pred_positions = pred[:, :, :2]
        target_positions = target[:, :, :2]

        pred_vis = pred[:, :, 2]
        target_vis = target[:, :, 2]

        pos_loss = self.mse_loss(pred_positions, target_positions)
        vis_loss = self.cross_entropy_loss(pred_vis, target_vis)

        dfl_loss = self.position_loss_weight * pos_loss + (1 - self.position_loss_weight) * vis_loss

        return dfl_loss, {
            "pos_loss": pos_loss,
            "vis_loss": vis_loss
        }