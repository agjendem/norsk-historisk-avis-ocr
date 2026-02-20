"""Shared column splitting logic for multi-column newspaper page images."""


def _detect_header_boundary(gray_pixels, width, height, threshold=200):
    """Find the y-coordinate where newspaper columns begin.

    Scans rows top-down looking for the first row where the vertical projection
    shows multiple distinct text regions separated by gaps — indicating the
    start of multi-column layout.

    Returns the y-coordinate of the column start, or 0 if no header detected.
    """
    # Minimum gap width (px) to count as a column separator
    min_gap = 15
    # Minimum number of column regions to consider it "multi-column"
    min_columns = 2
    # Scan in bands of rows to smooth out noise
    band_height = 20

    for y_start in range(0, height - band_height, band_height):
        y_end = min(y_start + band_height, height)
        # Count dark pixels per x-coordinate in this band
        x_dark = [0] * width
        for y in range(y_start, y_end):
            for x in range(width):
                if gray_pixels[x, y] < threshold:
                    x_dark[x] += 1

        # Classify each x as "has text" or "gap" in this band
        band_rows = y_end - y_start
        gap_threshold = band_rows * 0.01  # <1% dark pixels = gap
        in_text = False
        regions = 0
        gap_width = 0

        for x in range(width):
            if x_dark[x] > gap_threshold:
                if not in_text:
                    if gap_width >= min_gap or regions == 0:
                        regions += 1
                    in_text = True
                gap_width = 0
            else:
                in_text = False
                gap_width += 1

        if regions >= min_columns:
            # Found multi-column layout — header ends here
            # Go back a bit to avoid cutting into the first column row
            return max(0, y_start)

    return 0


def _detect_title_region(image, boundaries, threshold=200):
    """Detect a title region at the top of the page that spans multiple columns.

    After column boundaries are known, analyzes each column strip to find where
    regular body text begins. Title regions are characterized by large vertical
    gaps (>= large_gap_min px of blank rows) between lines of large-font text,
    while body text is dense and continuous.

    Args:
        image: PIL Image of the full page (grayscale or RGB).
        boundaries: Sorted list of column boundary x-coordinates (including 0
            and width).
        threshold: Grayscale value below which a pixel is considered dark.

    Returns:
        (title_image_or_none, body_top_per_column):
        - title_image: Cropped PIL Image of the title region, or None if no
          title detected.
        - body_top_per_column: List of y-coordinates (one per column) where
          body text begins. 0 for columns without a title region above them.
    """
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    n_cols = len(boundaries) - 1

    if n_cols < 2:
        return None, [0] * max(n_cols, 1)

    # Minimum run of blank rows to count as a "large gap" (title-like spacing).
    # At 300 DPI, body text has ~0px gaps between lines, while title text has
    # 40-80px gaps between large-font lines.
    large_gap_min = 40
    # Only scan the top portion of the page for title content
    scan_limit = int(height * 0.40)
    # A row is "blank" if fewer than 1% of its pixels are dark
    blank_frac = 0.01

    body_start_y = []

    for col_idx in range(n_cols):
        x_left = boundaries[col_idx]
        x_right = boundaries[col_idx + 1]
        col_width = x_right - x_left

        if col_width < 30:
            body_start_y.append(0)
            continue

        # Scan rows in the top region, find large gaps (runs of blank rows)
        gap_start = None
        last_large_gap_end = 0

        for y in range(scan_limit):
            count = 0
            for x in range(x_left, x_right):
                if pixels[x, y] < threshold:
                    count += 1
            is_blank = count < col_width * blank_frac

            if is_blank:
                if gap_start is None:
                    gap_start = y
            else:
                if gap_start is not None:
                    gap_len = y - gap_start
                    if gap_len >= large_gap_min:
                        last_large_gap_end = y
                    gap_start = None

        # body_start_y = row after the last large gap, or 0 if no large gaps
        body_start_y.append(last_large_gap_end)

    # Determine if there's a title region: columns with non-zero body_start_y
    # have title-like content at the top.
    # Require body_start_y > 5% of page height to filter noise.
    min_title_height = int(height * 0.05)
    elevated = [i for i in range(n_cols) if body_start_y[i] > min_title_height]

    if not elevated:
        return None, [0] * n_cols

    # Group into contiguous runs of adjacent columns
    groups = []
    current_group = [elevated[0]]
    for i in range(1, len(elevated)):
        if elevated[i] == elevated[i - 1] + 1:
            current_group.append(elevated[i])
        else:
            groups.append(current_group)
            current_group = [elevated[i]]
    groups.append(current_group)

    # Pick the largest group (or leftmost if tied)
    title_cols = max(groups, key=len)

    if not title_cols:
        return None, [0] * n_cols

    # Title region: from top of page to the max body_start_y across title columns,
    # spanning from left edge of first title column to right edge of last
    title_bottom = max(body_start_y[c] for c in title_cols)
    title_left = boundaries[title_cols[0]]
    title_right = boundaries[title_cols[-1] + 1]

    # Crop the title image
    title_image = image.crop((title_left, 0, title_right, title_bottom))

    # Build per-column body_top: title columns use their body_start_y,
    # non-title columns use 0
    body_top_per_col = [0] * n_cols
    for c in title_cols:
        body_top_per_col[c] = body_start_y[c]

    return title_image, body_top_per_col


def _find_gap_boundaries(gray_pixels, x_start, x_end, y_start, y_end,
                         expected_col_width, threshold=200, min_gap_px=8,
                         min_coverage=0.55):
    """Find column boundaries in a wide segment using gap coverage analysis.

    Computes a gap coverage profile: for each x-position, the fraction of rows
    where x is inside a run of consecutive light pixels (>=threshold) at least
    min_gap_px wide.  Then uses the expected column width to guide a search for
    boundaries at positions with maximum gap coverage.

    Args:
        gray_pixels: Pixel access object from a grayscale PIL Image.
        x_start, x_end: Horizontal extent of the segment (absolute coords).
        y_start, y_end: Vertical extent of the segment.
        expected_col_width: Median column width from Phase 1 (guides search).
        threshold: Grayscale value above which a pixel is considered light.
        min_gap_px: Minimum run length to count as a gap (filters inter-word spaces).
        min_coverage: Minimum gap coverage fraction to accept a boundary.

    Returns list of absolute x-coordinates for detected boundaries.
    """
    seg_width = x_end - x_start
    height = y_end - y_start
    if seg_width < 50 or height < 50:
        return []

    # Build gap coverage profile: for each relative x, count rows where x is
    # inside a gap run of >= min_gap_px consecutive light pixels.
    coverage = [0] * seg_width
    for y in range(y_start, y_end):
        run_start = None
        for rx in range(seg_width):
            ax = x_start + rx
            if gray_pixels[ax, y] >= threshold:
                if run_start is None:
                    run_start = rx
            else:
                if run_start is not None:
                    if rx - run_start >= min_gap_px:
                        for j in range(run_start, rx):
                            coverage[j] += 1
                    run_start = None
        if run_start is not None:
            if seg_width - run_start >= min_gap_px:
                for j in range(run_start, seg_width):
                    coverage[j] += 1

    # Smooth with 15px moving average
    half_w = 7
    smoothed = [0.0] * seg_width
    for i in range(seg_width):
        lo = max(0, i - half_w)
        hi = min(seg_width, i + half_w + 1)
        smoothed[i] = sum(coverage[lo:hi]) / (hi - lo) / height

    # Determine expected number of sub-columns and search for boundaries
    n_expected = round(seg_width / expected_col_width)
    if n_expected < 2:
        return []

    search_radius = int(expected_col_width * 0.3)
    boundaries = []
    for b in range(1, n_expected):
        expected_rx = int(b * seg_width / n_expected)
        lo = max(50, expected_rx - search_radius)
        hi = min(seg_width - 50, expected_rx + search_radius)
        if lo >= hi:
            continue

        best_rx = lo
        best_val = smoothed[lo]
        for rx in range(lo + 1, hi + 1):
            if smoothed[rx] > best_val:
                best_val = smoothed[rx]
                best_rx = rx

        if best_val >= min_coverage:
            boundaries.append(x_start + best_rx)

    return boundaries


def _save_debug_images(image, boundaries, debug_dir, body_top_per_col=None,
                       overlap_px=0, title_image=None):
    """Save annotated page image, column crops, and detection info."""
    from PIL import Image, ImageDraw, ImageFont

    debug_dir.mkdir(parents=True, exist_ok=True)
    width, height = image.size
    n_cols = len(boundaries) - 1

    if body_top_per_col is None:
        body_top_per_col = [0] * n_cols

    # Annotated full page
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for i, bx in enumerate(boundaries):
        if 0 < bx < width:
            draw.line([(bx, 0), (bx, height)], fill="blue", width=2)

    # Draw title bounding box if detected (semi-transparent green fill + thick outline)
    title_cols = []
    if title_image is not None:
        title_cols = [i for i in range(n_cols) if body_top_per_col[i] > 0]
        if title_cols:
            title_left = boundaries[title_cols[0]]
            title_right = boundaries[title_cols[-1] + 1]
            title_bottom = max(body_top_per_col[c] for c in title_cols)
            # Green shaded overlay for title region
            title_overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
            title_overlay_draw = ImageDraw.Draw(title_overlay)
            title_overlay_draw.rectangle(
                [(title_left, 0), (title_right, title_bottom)],
                fill=(0, 200, 0, 50),
            )
            annotated = Image.alpha_composite(annotated.convert("RGBA"), title_overlay)
            annotated = annotated.convert("RGB")
            draw = ImageDraw.Draw(annotated)
            # Thick green outline
            draw.rectangle(
                [(title_left, 0), (title_right, title_bottom)],
                outline=(0, 200, 0), width=5,
            )
            draw.text((title_left + 10, 8), "TITLE", fill=(0, 200, 0))

    # Draw per-column body_top lines (thick dashed-style green)
    for i in range(n_cols):
        if body_top_per_col[i] > 0 and i not in title_cols:
            left = boundaries[i]
            right = boundaries[i + 1]
            y = body_top_per_col[i]
            draw.line([(left, y), (right, y)], fill=(0, 200, 0), width=4)

    # Draw overlap regions as semi-transparent red shading
    if overlap_px > 0:
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        for bx in boundaries:
            if 0 < bx < width:
                ol = max(0, bx - overlap_px)
                or_ = min(width, bx + overlap_px)
                overlay_draw.rectangle(
                    [(ol, 0), (or_, height)],
                    fill=(255, 0, 0, 40),
                )
        annotated = Image.alpha_composite(annotated.convert("RGBA"), overlay)
        annotated = annotated.convert("RGB")
    # Label columns
    draw = ImageDraw.Draw(annotated)
    for i in range(n_cols):
        cx = (boundaries[i] + boundaries[i + 1]) // 2
        label_y = body_top_per_col[i] + 10 if body_top_per_col[i] > 0 else 10
        draw.text((cx - 10, label_y), str(i + 1), fill="blue")
    annotated.save(debug_dir / "page_annotated.png")

    # Save title crop if detected
    if title_image is not None:
        title_image.save(debug_dir / "title_crop.png")

    # Column crops (with overlap padding matching OCR crops)
    for i in range(n_cols):
        left = max(0, boundaries[i] - overlap_px)
        right = min(width, boundaries[i + 1] + overlap_px)
        if right - left < 30:
            continue
        col_body_top = body_top_per_col[i] if i < len(body_top_per_col) else 0
        col_img = image.crop((left, col_body_top, right, height))
        col_img.save(debug_dir / f"column_{i + 1}_crop.png")

    # Detection info
    info_lines = [
        f"Image size: {width} x {height}",
        f"Overlap padding: {overlap_px}px",
        f"Boundaries: {boundaries}",
        f"Columns: {n_cols}",
    ]

    # Title detection info
    if title_image is not None:
        title_cols = [i for i in range(n_cols) if body_top_per_col[i] > 0]
        title_bottom = max(body_top_per_col[c] for c in title_cols)
        title_left = boundaries[title_cols[0]]
        title_right = boundaries[title_cols[-1] + 1]
        info_lines.append(
            f"Title detected: columns {[c + 1 for c in title_cols]}, "
            f"x={title_left}-{title_right}, y=0-{title_bottom}"
        )
        info_lines.append(f"Title crop size: {title_image.size[0]} x {title_image.size[1]}")
    else:
        info_lines.append("Title detected: none")

    info_lines.append(f"Body top per column: {body_top_per_col}")
    info_lines.append("")

    for i in range(n_cols):
        w = boundaries[i + 1] - boundaries[i]
        pad_left = min(overlap_px, boundaries[i])
        pad_right = min(overlap_px, width - boundaries[i + 1])
        crop_w = w + pad_left + pad_right
        col_body_top = body_top_per_col[i] if i < len(body_top_per_col) else 0
        info_lines.append(
            f"  Column {i + 1}: x={boundaries[i]}-{boundaries[i + 1]}, "
            f"width={w}px, crop={crop_w}px (pad L={pad_left} R={pad_right}), "
            f"body_top={col_body_top}"
        )
    (debug_dir / "detection_info.txt").write_text("\n".join(info_lines) + "\n", encoding="utf-8")


def _split_columns(image, debug_dir=None, overlap_px=20):
    """Split a newspaper page image into individual column images.

    Uses a three-phase algorithm:
    1. Detect ink divider lines via vertical projection profile
    2. Subdivide wide segments using row-by-row gap voting
    3. Merge boundaries, crop columns with overlap padding

    Args:
        image: PIL Image of the full page.
        debug_dir: Optional directory to save debug images.
        overlap_px: Pixels of padding to add on each side of every column
            crop. Compensates for non-linear scan distortion that shifts
            gutter positions across the page height. Default 20px.

    Returns (title_image_or_none, [column_images]).
    Title detection finds large-font title regions spanning multiple columns
    and excludes them from column crops. Column crops for title columns start
    at the per-column body_start_y instead of 0.
    If only one column is detected, returns (None, [original_image]).
    """
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()

    body_top = 0

    # Phase 1: detect ink divider lines via vertical projection profile
    v_profile = [0] * width
    for x in range(width):
        for y in range(body_top, height):
            if pixels[x, y] < 200:
                v_profile[x] += 1

    body_height = height - body_top
    divider_threshold = body_height * 0.8
    divider_xs = []
    in_divider = False
    div_start = 0
    for x in range(width):
        if v_profile[x] >= divider_threshold:
            if not in_divider:
                div_start = x
                in_divider = True
        else:
            if in_divider:
                divider_xs.append((div_start + x) // 2)
                in_divider = False
    if in_divider:
        divider_xs.append((div_start + width - 1) // 2)

    # Phase 2: subdivide wide segments using gap voting
    phase1_boundaries = [0] + divider_xs + [width]

    # Compute median segment width from Phase 1
    seg_widths = [phase1_boundaries[i + 1] - phase1_boundaries[i]
                  for i in range(len(phase1_boundaries) - 1)]
    seg_widths_sorted = sorted(seg_widths)
    if seg_widths_sorted:
        mid = len(seg_widths_sorted) // 2
        median_width = seg_widths_sorted[mid]
    else:
        median_width = width

    all_boundaries = set(phase1_boundaries)

    for i in range(len(phase1_boundaries) - 1):
        seg_left = phase1_boundaries[i]
        seg_right = phase1_boundaries[i + 1]
        seg_w = seg_right - seg_left

        # Only subdivide segments wider than 1.5x the median
        if seg_w > median_width * 1.5:
            gap_bounds = _find_gap_boundaries(
                pixels, seg_left, seg_right, body_top, height,
                expected_col_width=median_width)
            all_boundaries.update(gap_bounds)

    # Phase 3: merge, sort, and crop
    boundaries = sorted(all_boundaries)

    # Filter out narrow segments to get final boundaries
    final_boundaries = [boundaries[0]]
    for i in range(len(boundaries) - 1):
        left = boundaries[i]
        right = boundaries[i + 1]
        if right - left < 30:
            continue
        final_boundaries.append(right)

    # Detect title region using final column boundaries
    title_image, body_top_per_col = _detect_title_region(
        image, final_boundaries)

    # Crop column images with per-column body_top
    columns = []
    n_cols = len(final_boundaries) - 1
    for i in range(n_cols):
        left = final_boundaries[i]
        right = final_boundaries[i + 1]
        # Apply overlap padding, clamped to image bounds
        crop_left = max(0, left - overlap_px)
        crop_right = min(width, right + overlap_px)
        col_body_top = body_top_per_col[i] if i < len(body_top_per_col) else 0
        col_img = image.crop((crop_left, col_body_top, crop_right, height))
        columns.append(col_img)

    if debug_dir:
        _save_debug_images(image, final_boundaries, debug_dir,
                           body_top_per_col=body_top_per_col,
                           overlap_px=overlap_px, title_image=title_image)

    if not columns:
        return (None, [image])

    return (title_image, columns)
