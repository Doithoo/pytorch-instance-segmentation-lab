# Dataset Format

The built-in provider expects `PennFudanPed/PNGImages/<id>.png` and `PennFudanPed/PedMasks/<id>_mask.png`. Mask value 0 is background; every positive integer is one person instance. The target uses float32 half-open `xyxy` boxes and bool masks. Do not provide a semantic class map where touching people share one value.
