ax = axes[row, col]
ax.imshow(img)
ax.axis("off")

# --- correctness per predictor (using your GT class name) ---
# if you have auto_map_pred(...) use that; otherwise simple equality is fine
pre_mapped = auto_map_pred(pre_label, GT_CLASSES)
ft_mapped  = auto_map_pred(ft_label,  GT_CLASSES)

pre_color = "green" if pre_mapped == cls else "red"
ft_color  = "green" if ft_mapped  == cls else "red"
gt_color  = "blue"   # GT always same color

# --- put text *below* each image, left-aligned, different colors ---
ax.text(
    0.0, -0.05,
    f"GT : {cls}",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=8,
    color=gt_color,
    clip_on=False,
)

ax.text(
    0.0, -0.15,
    f"Pre: {pre_label} ({pre_prob:.2f})",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=7,
    color=pre_color,
    clip_on=False,
)

ax.text(
    0.0, -0.25,
    f"FT : {ft_label} ({ft_prob:.2f})",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=7,
    color=ft_color,
    clip_on=False,
)
