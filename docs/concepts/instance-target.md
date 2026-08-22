# Instance Target

`target` has aligned first dimension `N`: `boxes[N,4]`, foreground `labels[N]`, `masks[N,H,W]`, `area[N]`, `iscrowd[N]`, and one `image_id[1]`. Boxes are derived from masks after every geometric transform; empty masks are filtered with all aligned fields.
