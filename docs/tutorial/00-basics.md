# Instance Segmentation Basics

Object detection returns boxes. Semantic segmentation returns one class per pixel. Instance segmentation returns a variable-length list of independent objects; each object has a class, a half-open `xyxy` box, a binary mask, and a score. Two people must remain two masks even when their pixels touch.
