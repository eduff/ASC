# Additive Signal Change Analysis tool

## Setup environment
```bash
>> source <dhcpdir>/setup.py
```

## Usage
```bash
usage: pipeline.py [-h] --func FUNC --struct STRUCT --age AGE
                   [--func_acqp FUNC_ACQP] [--func_sliceorder FUNC_SLICEORDER]
                   [--sbref SBREF] [--fix_data FIX_DATA]
                   [--fmap_phasediff FMAP_PHASEDIFF] [--fmap_mag FMAP_MAG]
                   [--b0 B0] [--b0_acqp B0_ACQP]
                   [--b0_echospacing B0_ECHOSPACING] [--b0_pedir B0_PEDIR]
                   [--func_echospacing FUNC_ECHOSPACING]
                   [--func_pedir FUNC_PEDIR]
                   [--sbref_echospacing SBREF_ECHOSPACING]
                   [--sbref_pedir SBREF_PEDIR] [--mbs]
                   [--left_midsurface LEFT_MIDSURFACE]
                   [--right_midsurface RIGHT_MIDSURFACE] [--seg SEG]
                   [--workdir WORKDIR] [--run]
```

## Examples

### Basic

Key (required) args:
```bash
--func      (nifti)
--struct    (nifti)
--age       (int or float)
```
If age is a float it will be rounded to nearest int

Example:
```bash
${DHCPDIR}/dhcp/func/pipeline.py \
    --func=func.nii.gz \
	--struct=struct.nii.gz \
	--age=43
```

* Segmentation of T2 with DrawEM
* Distortion (susceptibility) correction = None
* Motion correction = `mcflirt` volume-to-volume
* Registration = `flirt` BBR func-to-struct
* ICA = True
* Denoising = `fix` extract features but not classification

### Basic (without segmentation)

Key (required) args:
```bash
--func      (nifti)
--struct    (nifti)
--seg       (nifti)
--age       (int or float)
```
If age is a float it will be rounded to nearest int

Example:
```bash
${DHCPDIR}/dhcp/func/pipeline.py \
    --func=func.nii.gz \
	--struct=struct.nii.gz \
	--struct_seg=struct_seg.nii.gz \
	--age=43
```

* Distortion (susceptibilty) correction = None
* Motion correction = `mcflirt` volume-to-volume
* Registration = `flirt` BBR func-to-struct
* ICA = True
* Denoising = `fix` extract features but not classification

### SBREF

Key args:
```bash
--sbref (nifti)
```

Example:
```bash
${DHCPDIR}/dhcp/func/pipeline.py \
    --func=func.nii.gz \
    --sbref=sbref.nii.gz \
	--struct=struct.nii.gz \
	--struct_seg=struct_seg.nii.gz \
	--age=43
```

* Distortion (susceptibilty) correction = None
* Motion correction = `mcflirt` volume-to-volume
* Registration = `flirt` 6DOF func-to-sbref + `flirt` BBR sbref-to-struct
* ICA = True
* Denoising = `fix` extract features but not classification

### Acquisition parameters

Key args:
```bash
--func_echospacing      (float)
--func_pedir            (AP/PA/LR/RL/IS/SI)
--sbref_echospacing     (float)
--sbref_pedir           (AP/PA/LR/RL/IS/SI)
```
If `--sbref_echospacing` or `--sbref_pedir` are not provided then the equivalent `func` values will be used.

Example:
```bash
${DHCPDIR}/dhcp/func/pipeline.py \
    --func=func.nii.gz \
    --func_echospacing=0.0007216 \  
    --func_pedir=PA \
    --sbref=sbref.nii.gz \
    --sbref_echospacing=0.0007216 \  
    --sbref_pedir=PA \
	--struct=struct.nii.gz \
	--struct_seg=struct_seg.nii.gz \
	--age=43
```

* Distortion (susceptibility) correction = None
* Motion correction = `eddy` volume-to-volume
* Registration = `flirt` 6DOF func-to-sbref + `flirt` BBR sbref-to-struct
* ICA = True
* Denoising = `fix` extract features but not classification

### Slice-to-volume motion correction

Key args:
```bash
--func_echospacing  (float)
--func_pedir        (AP/PA/LR/RL/IS/SI)
--func_sliceorder   (txt file)

```

Example:
```bash
${DHCPDIR}/dhcp/func/pipeline.py \
    --func=func.nii.gz \
    --func_echospacing=0.0007216 \  
    --func_pedir=PA \
    --func_sliceorder=func.slorder \
    --sbref=sbref.nii.gz \
	--struct=struct.nii.gz \
	--struct_seg=struct_seg.nii.gz \
	--age=43
```

* Distortion (susceptibility) correction = None
* Motion correction = `eddy` slice-to-volume
* Registration = `flirt` 6DOF func-to-sbref + `flirt` BBR sbref-to-struct
* ICA = True
* Denoising = `fix` extract features but not classification

### Distortion (susceptibility) correction

Key args:
```bash
--func_echospacing      (float)
--func_pedir            (AP/PA/LR/RL/IS/SI)

    AND

--fmap_phasediff        (nifti)
--fieldmap_magnitude    (nifti)

    OR

--b0                    (nifti)
--b0_echospacing        (float)
--b0_pedir              (AP/PA/LR/RL/IS/SI)

```

Example:
```bash
${DHCPDIR}/dhcp/func/pipeline.py \
    --func=func.nii.gz \
    --func_echospacing=0.0007216 \  
    --func_pedir=PA \
    --func_sliceorder=func.slorder \
    --sbref=sbref.nii.gz \
	--struct=struct.nii.gz \
	--struct_seg=struct_seg.nii.gz \
    --b0=b0.nii.gz \
    --b0_pedir=AP,AP,PA,PA \
    --b0_echospacing=0.0007188 \
	--age=43
```

* Fieldmaps estimated from B0's using `topup`
* Distortion (susceptibility) correction
    * `eddy` distortion correction of func
    * `flirt` distorion correction of sbref (as part of BBR)
* Motion correction = `eddy` slice-to-volume
* Registration = `flirt` 6DOF func-to-sbref + `flirt` BBR sbref-to-struct
* ICA = True
* Denoising = `fix` extract features but not classification

### Motion-by-susceptibility

Key args:
```bash
--mbs
```

Example:
```bash
${DHCPDIR}/dhcp/func/pipeline.py \
    --func=func.nii.gz \
    --func_echospacing=0.0007216 \  
    --func_pedir=PA \
    --func_sliceorder=func.slorder \
    --sbref=sbref.nii.gz \
	--struct=struct.nii.gz \
	--struct_seg=struct_seg.nii.gz \
    --b0=b0.nii.gz \
    --b0_pedir=AP,AP,PA,PA \
    --b0_echospacing=0.0007188 \
    --mbs \
	--age=43
```

* Fieldmaps estimated from B0's using `topup`
* Distortion (susceptibility) correction
    * `eddy` motion-by-susceptibility distortion correction of func
    * `flirt` distorion correction of sbref (as part of BBR)
* Motion correction = `eddy` slice-to-volume
* Registration = `flirt` 6DOF func-to-sbref + `flirt` BBR sbref-to-struct
* ICA = True
* Denoising = `fix` extract features but not classification

### Denoise

Key args:
```bash
--fixdata (RData)
```

Example:
```bash
${DHCPDIR}/dhcp/func/pipeline.py \
    --func=func.nii.gz \
    --func_echospacing=0.0007216 \  
    --func_pedir=PA \
    --func_sliceorder=func.slorder \
    --sbref=sbref.nii.gz \
	--struct=struct.nii.gz \
	--struct_seg=struct_seg.nii.gz \
    --b0=b0.nii.gz \
    --b0_pedir=AP,AP,PA,PA \
    --b0_echospacing=0.0007188 \
    --fixdata=trained.RData
	--age=43
```

* Fieldmaps estimated from B0's using `topup`
* Distortion (susceptibility) correction
    * `eddy` motion-by-susceptibility distortion correction of func
    * `flirt` distorion correction of sbref (as part of BBR)
* Motion correction = `eddy` slice-to-volume
* Registration = `flirt` 6DOF func-to-sbref + `flirt` BBR sbref-to-struct
* ICA = True
* Denoising = `fix` extract features and `fix` classification

### Sample-to-surface

Key args:
```bash
--left_midsurface   (gifti)
--right_midsurface  (gifti)
```

Example:
```bash
${DHCPDIR}/dhcp/func/pipeline.py \
    --func=func.nii.gz \
    --func_echospacing=0.0007216 \  
    --func_pedir=PA \
    --func_sliceorder=func.slorder \
    --sbref=sbref.nii.gz \
	--struct=struct.nii.gz \
	--struct_seg=struct_seg.nii.gz \
    --b0=b0.nii.gz \
    --b0_pedir=AP,AP,PA,PA \
    --b0_echospacing=0.0007188 \
    --fixdata=trained.RData
	--age=43 \
    --left_midsurface=L.midthickness.native.surf.gii \
    --right_midsurface=R.midthickness.native.surf.gii
```

* Fieldmaps estimated from B0's using `topup`
* Distortion (susceptibility) correction
    * `eddy` distortion correction of func
    * `flirt` distorion correction of sbref (as part of BBR)
* Motion correction = `eddy` slice-to-volume
* Registration = `flirt` 6DOF func-to-sbref + `flirt` BBR sbref-to-struct
* ICA = True
* Denoising = `fix` extract features and `fix` classification
* func sampled to mid-thickeness surface
