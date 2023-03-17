from monai.networks.nets import UNet
from monai.networks.layers import Norm

def build_network(args, device):
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