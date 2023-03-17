from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    EnsureTyped,
    EnsureType,
    ConcatItemsd,
    ToTensord,
    RandAffined,
    RandGaussianNoised,
    RandShiftIntensity,
    RandSpatialCropSamplesd,
)
from monai.data import list_data_collate, DataLoader, Dataset
import sys
sys.path.insert(0, '/media/sambit/Seagate Expansion Drive/G/Uppsala University/Courses/Semesters/2023/1_January_July/Advanced Scientific Programming with Python/Project/Advanced_Scientific_Programming_with_Python_Project')

def data_transform(args, purpose):
	train_transforms = Compose(
	    [
	        LoadImaged(keys=["SUV", "CT", "SEG"]),
	        AddChanneld(keys=["SUV", "CT", "SEG"]),
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
	            spatial_size=args.roi_size,
	            pos=args.pos,
	            neg=args.neg,
	            num_samples=args.num_samples,
	            #image_key="image",
	            image_threshold=0,
	        ),
	        ConcatItemsd(keys=["SUV", "CT"], name="PET_CT", dim=0),# concatenate pet and ct channels
	        ToTensord(keys=["PET_CT", "SEG"]),
	    ]
	)
	val_transforms = Compose(
	    [
	        LoadImaged(keys=["SUV", "CT", "SEG"]),
	        AddChanneld(keys=["SUV", "CT", "SEG"]),
	        ScaleIntensityRanged(
	            keys=["CT"], a_min=args.CT_min, a_max=args.CT_max,
	            b_min=0.0, b_max=1.0, clip=True,
	        ),
	        ScaleIntensityRanged(
	            keys=["SUV"], a_min=args.SUV_min, a_max=args.SUV_max,
	            b_min=0.0, b_max=1.0, clip=True,
	        ),
	        CropForegroundd(keys=["SUV", "CT", "SEG"], source_key="CT"),
	        ConcatItemsd(keys=["SUV", "CT"], name="PET_CT", dim=0),# concatenate pet and ct channels
	        ToTensord(keys=["PET_CT", "SEG"]),
	    ]
	)
	if purpose == "TRAIN":
		return train_transforms
	elif purpose == "VALIDATION":
		return val_transforms

def prepare_data(args, df_train, df_val):
	CT_train = sorted(df_train['CT'].tolist())
	SUV_train = sorted(df_train['SUV'].tolist())
	SEG_train = sorted(df_train['SEG'].tolist())

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
	train_transforms = data_transform(args, "TRAIN")
	val_transforms = data_transform(args, "VALIDATION")

	train_dset = Dataset(data=train_files, transform=train_transforms)
	train_loader = DataLoader(train_dset, batch_size=args.batch_size_train, num_workers=args.num_workers, collate_fn = list_data_collate)

	val_dset = Dataset(data=val_files, transform=val_transforms)
	val_loader = DataLoader(val_dset, batch_size=args.batch_size_val, num_workers=args.num_workers, collate_fn = list_data_collate)
	
	return train_loader, val_loader, val_files
