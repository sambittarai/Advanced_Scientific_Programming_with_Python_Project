from monai.networks.nets import UNet
from monai.networks.layers import Norm

def build_network(args, device):
    """
    Builds a UNet model for image segmentation.

    Args:
        args (argparse.Namespace): A namespace containing the command-line arguments.
            - dimensions (int): The number of dimensions of the UNet. Either 2 or 3.
            - in_channels (int): The number of channels in the input image.
            - out_channels (int): The number of channels in the output segmentation mask.
            - kernel_channels (int): The number of channels in the intermediate feature maps.
            - strides (tuple[int]): The downsampling factor for each level of the UNet.
            - residual_units (int): The number of residual units in each level of the UNet.
            - dropout (float): The dropout probability.

        device (torch.device): The device on which to create the model.

    Returns:
        torch.nn.Module: A UNet model for image segmentation.

    """
	model = UNet(
				dimensions=args.dimensions,
				in_channels=args.in_channels,
				out_channels=args.out_channels,
				channels=args.kernel_channels,
				strides=args.strides,
				num_res_units=args.residual_units,
				norm=Norm.BATCH,
				dropout=args.dropout,
			).to(device)
	return model