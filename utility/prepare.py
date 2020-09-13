from glob import glob
import pydicom as dicom
import dicom_numpy
import numpy as np
import SimpleITK as sitk


def _get_array_from_dicom_file(list_of_dicom_files):
    datasets = [dicom.read_file(f) for f in list_of_dicom_files]
    try:
        voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(datasets)
    except dicom_numpy.DicomImportException as e:
        # invalid DICOM data
        raise
    return voxel_ndarray


def _get_array_from_mhd_raw_file(path):
    ds = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(ds)
    return image


def save_luna_file():
    path = f'''../sample/1.3.6.1.4.1.14519.5.2.1.6279.6001.272042302501586336192628818865.mhd'''
    img = _get_array_from_mhd_raw_file(path=path)
    np.save('../resources/2d_img.npy', img[140, :, :])


def save_my_chest_ct():
    img = _get_array_from_dicom_file(glob('../resources/dicoms/SR_4/*.dcm'))
    img = np.moveaxis(img, [0, 1, 2], [2, 1, 0])
    np.save('../resources/my_lungs.npy', img[90, :, :])


if __name__ == '__main__':
    save_my_chest_ct()
    save_luna_file()
