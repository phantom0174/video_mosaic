# Video Mosaic

## 🔧 Developers

- @phantom0174
- [@yuchenleeBB][bb]

## 📜 Overview

> 🎬 Demo: https://www.youtube.com/watch?v=Dh6Jhp5Qp9U

This repository is the public release of the final project for the **NTHU Parallel Programming** course.  
It is designed to significantly accelerate the original [photo-mosaic][photo_mosaic] implementation, making it feasible to generate photo mosaics for **every frame of a video** within a reasonable amount of time.

Additional details can be found in the [project slides][slides_link].

## 📁 Project Structure

```
.
├── src/common.h          # shared configuration
├── src/opt_turbo.cu      # CUDA version (migrated from HIP version)
├── src/opt_turbo.cpp     # HIP version
├── src/make_cache.cpp    # preprocess tile library
├── tools/sewing.txt      # ffmpeg command for stitching frames
├── tools/process_pdf.py  # preprocess Epstein pdf files into images
├── bad_frames.zip        # Bad Apple frame DB
```

## 🔢 Performance

### Matching Setting

- target: Bad Apple (6571 frames)
- tile library: Bad Apple frames
- resolution: 40x40
- output scale: 3.0 (1440*1080)

### Results

| version      | system spec                    | core matching time | equivalent speedup | comment         |
|--------------|--------------------------------|--------------------|--------------------|-----------------|
| photo-mosaic | TR 7960X w/ 2x 5070 Ti 16G     | ~10 mins.          | baseline           |                 |
| HIP          | Xeon X5670 w/ 2x MI100         | 33 secs.           | x18.2              | I/O bound (CPU) |
| CUDA (opt)   | Ultra 7 265K w/ 1x 5060 Ti 16G | 6.5 secs.          | x184.6             |                 |

## 🧱 Requirements

### CUDA Platform (Primary)

- Host OS: Windows 11
- Linux Environment: WSL2
- Kernel: 6.6.87.2-microsoft-standard-WSL2
- Architecture: x86_64
- GPU: NVIDIA RTX 5060 Ti 16GB
- NVIDIA Driver: supports CUDA 13.1
- CUDA Runtime: 13.1

### HIP Platform

- OS: Debian GNU/Linux
- Kernel: 6.1.0-39-amd64
- Architecture: x86_64
- GPU: AMD MI100
- ROCm: unavailable (currently offline)

## ☘️ Usage

> frame extraction for target videos (`get_frames.py`) can be found in [photo-mosaic][photo_mosaic]

1. Get the essential tools:
    > run `get_ffmpeg.sh`, `get_img_turbo.sh`
2. Tune matching parameters in `common.h`
3. Build the cache generator:
    > make make-cache
4. Build the executable:
    > make opt-turbo (hip) / opt-cuda (cuda)
5. Generate the tile cache:
    > run `make-cache`
6. Run the mosaic generator:
    > run `opt-turbo`/`opt-cuda`
7. Stitch frames into a video using the command in `sewing.txt`
8. Done.


[bb]: https://github.com/yuchenleeBB
[slides_link]: https://docs.google.com/presentation/d/1T5pksOd2U5jfdyx7xXKntr_oqUi2p-tagPI_H8_0BZI/edit?usp=sharing
[photo_mosaic]: https://github.com/phantom0174/photo-mosaics