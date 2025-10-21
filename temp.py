def scan_pairs(data_root: Path, images_glob: str = "*.jpg"):
    """
    Pairs images under .../images and JSONs under .../annotations by a
    normalized stem, handling names like '001.jpg' <-> '001.jpg.json'.
    """
    images_dir = data_root / "images"
    ann_dir    = data_root / "annotations"
    if not images_dir.is_dir():
        raise SystemExit(f"images/ not found under: {data_root}")
    if not ann_dir.is_dir():
        raise SystemExit(f"annotations/ not found under: {data_root}")

    # 1) collect images (case-insensitive, jpg/jpeg/png)
    pats = ["*.jpg","*.JPG","*.jpeg","*.JPEG","*.png","*.PNG"]
    images = []
    for pat in pats:
        images += list(images_dir.rglob(pat))

    # 2) collect annotations (exclude meta.json)
    annotations = [p for p in ann_dir.rglob("*.json") if p.name.lower() != "meta.json"]

    def norm_stem_for_image(p: Path) -> str:
        # '001.jpg' -> '001'
        return p.stem

    def norm_stem_for_anno(p: Path) -> str:
        """
        '001.json'       -> '001'
        '001.jpg.json'   -> '001'
        'foo.PNG.json'   -> 'foo'
        """
        name = p.name
        if name.lower().endswith(".json"):
            name = name[:-5]  # drop .json
        # strip a possible image extension from the remaining base
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        return name

    img_by = {norm_stem_for_image(p): p for p in images}
    ann_by = {norm_stem_for_anno(p):  p for p in annotations}

    common = sorted(set(img_by) & set(ann_by))
    pairs = [ImgAnnPair(img_by[s], ann_by[s]) for s in common]

    if not pairs:
        raise SystemExit(
            "No pairs found.\n"
            f" images_dir={images_dir.resolve()} (found {len(images)} files)\n"
            f" annotations_dir={ann_dir.resolve()} (found {len(annotations)} files)\n"
            f" example image stems: {list(img_by.keys())[:5]}\n"
            f" example anno stems : {list(ann_by.keys())[:5]}"
        )
    return pairs
