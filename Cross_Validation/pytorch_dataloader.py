from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    ConcatItemsd,
    RandAffined,
    ToTensord,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandShiftIntensity,
    RandSpatialCropSamplesd,
    DivisiblePadd
)
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset, SmartCacheDataset, pad_list_data_collate

import sys
sys.path.insert(0, '/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET')
from utils import get_df

def prepare_data(args, df_train, df_val):
	CT_train = sorted(df_train['CT'].tolist())
	SUV_train = sorted(df_train['SUV'].tolist())
	SEG_train = sorted(df_train['SEG'].tolist())

	#df_val = get_df(args.data_path_val)
	CT_val = sorted(df_val['CT'].tolist())
	SUV_val = sorted(df_val['SUV'].tolist())
	SEG_val = sorted(df_val['SEG'].tolist())

	train_files = [
		{"SUV": SUV_name, "CT": CT_name, "SEG": SEG_name}
		for SUV_name, CT_name, SEG_name in zip(SUV_train, CT_train, SEG_train)
	]
	val_files = [
		{"SUV": SUV_name, "CT": CT_name, "SEG": SEG_name}
		for SUV_name, CT_name, SEG_name in zip(SUV_val, CT_val, SEG_val)
	]

	# define transforms for image and segmentation
	train_transforms = Compose(
	    [
	        LoadImaged(keys=["SUV", "CT", "SEG"]),
	        AddChanneld(keys=["SUV", "CT", "SEG"]),
	        #EnsureChannelFirstd(keys=["SUV", "CT", "SEG"]),
	        ScaleIntensityRanged(
	            keys=["CT"], a_min=args.CT_min, a_max=args.CT_max,
	            b_min=0.0, b_max=1.0, clip=True,
	        ),
	        ScaleIntensityRanged(
	            keys=["SUV"], a_min=args.SUV_min, a_max=args.SUV_max,
	            b_min=0.0, b_max=1.0, clip=True,
	        ),
	        CropForegroundd(keys=["SUV", "CT", "SEG"], source_key="CT"),
	        RandCropByPosNegLabeld(
	            keys=["SUV", "CT", "SEG"],
	            label_key="SEG",
	            #spatial_size=(160, 160, 160),
	            spatial_size=args.roi_size,
	            pos=args.pos,
	            neg=args.neg,
	            num_samples=args.num_samples,
	            #image_key="image",
	            image_threshold=0,
	        ),
	        #DivisiblePadd(keys=["SUV", "CT", "SEG"], k=64, allow_missing_keys=False),

	        #Augmentations
	        #RandFlipd(keys=["SUV", "CT", "SEG"], prob=0.4, spatial_axis=0),
	        #RandFlipd(keys=["SUV", "CT", "SEG"], prob=0.4, spatial_axis=1),
	        #RandFlipd(keys=["SUV", "CT", "SEG"], prob=0.4, spatial_axis=2),
	        #RandAffined(keys=['SUV', "CT", 'SEG'], prob=0.3, translate_range=10),
	        #RandRotate90d(keys=["SUV", "CT", "SEG"], prob=0.4, spatial_axes=[0, 2]),
	        #RandGaussianNoised(keys=["SUV", "CT"], prob=0.2),
	        #RandShiftIntensityd(keys=["SUV"], offsets=(0,30), prob=0.2),

	        ConcatItemsd(keys=["SUV", "CT"], name="PET_CT", dim=0),# concatenate pet and ct channels

	        #EnsureTyped(keys=["image", "label"]),
	        #ToTensord(keys=["SUV", "SEG"]),
	        ToTensord(keys=["PET_CT", "SEG"]),
	    ]
	)

	val_transforms = Compose(
	    [
	        LoadImaged(keys=["SUV", "CT", "SEG"]),
	        AddChanneld(keys=["SUV", "CT", "SEG"]),
	        #EnsureChannelFirstd(keys=["SUV", "CT", "SEG"]),
	        ScaleIntensityRanged(
	            keys=["CT"], a_min=args.CT_min, a_max=args.CT_max,
	            b_min=0.0, b_max=1.0, clip=True,
	        ),
	        ScaleIntensityRanged(
	            keys=["SUV"], a_min=args.SUV_min, a_max=args.SUV_max,
	            b_min=0.0, b_max=1.0, clip=True,
	        ),
	        CropForegroundd(keys=["SUV", "CT", "SEG"], source_key="CT"),
	        #RandCropByPosNegLabeld(
	        #    keys=["SUV", "CT", "SEG"],
	        #    label_key="SEG",
	        #    spatial_size=(160, 160, 160),
	        #    pos=4,
	        #    neg=1,
	        #    num_samples=2,
	        #    #image_key="image",
	        #    image_threshold=0,
	        #),
	        ConcatItemsd(keys=["SUV", "CT"], name="PET_CT", dim=0),# concatenate pet and ct channels

	        #EnsureTyped(keys=["PET_CT", "SEG"]),
	        #ToTensord(keys=["SUV", "SEG"]),
	        ToTensord(keys=["PET_CT", "SEG"]),
	    ]
	)

	train_dset = Dataset(data=train_files, transform=train_transforms)
	train_loader = DataLoader(train_dset, batch_size=args.batch_size_train, num_workers=args.num_workers, collate_fn = list_data_collate)
	#train_loader = DataLoader(train_dset, batch_size=args.batch_size_train, num_workers=args.num_workers, collate_fn = pad_list_data_collate)

	val_dset = Dataset(data=val_files, transform=val_transforms)
	val_loader = DataLoader(val_dset, batch_size=args.batch_size_val, num_workers=args.num_workers, collate_fn = list_data_collate)
	#val_loader = DataLoader(val_dset, batch_size=args.batch_size_val, num_workers=args.num_workers, collate_fn = pad_list_data_collate)
	return train_loader, val_loader, val_files
