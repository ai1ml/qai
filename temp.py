ax = axes[row, col]
ax.imshow(img)
ax.axis("off")

# ---- Compute correctness ----
is_correct = (ft_label.lower() == cls.lower())
title_color = "green" if is_correct else "red"

# ---- Rich title with HTML-like formatting ----
title_text = (
    f"{cls} (GT)\n"
    f"Pre: {pre_label} ({pre_prob:.2f})\n"
    f"FT : {ft_label} ({ft_prob:.2f})"
)

# ---- Apply formatted title ----
ax.set_title(
    title_text,
    fontsize=8,
    loc="left",       # left-aligned
    color=title_color # green if correct, red if wrong
)
