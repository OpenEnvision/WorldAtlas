"""CPU applicability of refined masks after the executed image crop.

Only absent foreground/background in the *input conditioning frame* is N/A.
Malformed annotations and tracking/flow failures remain errors, never N/A.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from worldatlas.benchmark.mask_geometry import transform_mask
from worldatlas.benchmark.mask_provenance import mask_files, validate_mask_metadata

PROTOCOL='conditioning_mask_support_v1'


def conditioning_mask_support(mask_dir, *, source_path, relative_path, geometry):
    directory=Path(mask_dir)
    validate_mask_metadata(directory,relative_path)
    with Image.open(source_path) as source:
        if list(source.size)!=geometry.get('source_size'):
            raise ValueError('Mask geometry source dimensions differ')
    output=geometry.get('output_size',[])
    if len(output)!=2 or any(type(n) is not int or n<=0 for n in output):
        raise ValueError('Invalid conditioning output dimensions')
    union=np.zeros((output[1],output[0]),dtype=bool)
    files=mask_files(directory)
    if not files:
        raise ValueError('Manual-empty annotations must use the existing metadata N/A gate')
    areas=[];empty=[]
    for path in files:
        with Image.open(path) as image:
            raw=np.asarray(image.convert('L'))>0
        if not raw.any():
            raise ValueError(f'Saved annotation mask is empty before conditioning crop: {path}')
        mask=Image.fromarray(raw.astype(np.uint8)*255)
        aligned=np.asarray(transform_mask(mask,geometry,tuple(output)))>0
        area=int(aligned.sum());areas.append(area)
        if not area:empty.append(path.name)
        union|=aligned
    reason=None
    if not union.any():reason='conditioning_crop_has_no_foreground'
    elif union.all():reason='conditioning_crop_has_no_background'
    return dict(protocol=PROTOCOL,not_applicable_reason=reason,
                output_size=output,annotated_instances=len(files),
                visible_instances=sum(a>0 for a in areas),cropped_out_instances=empty,
                foreground_pixels_by_instance=areas,foreground_union_pixels=int(union.sum()),
                total_pixels=int(union.size))
