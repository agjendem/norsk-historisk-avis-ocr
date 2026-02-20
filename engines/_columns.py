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


def _save_debug_images(image, boundaries, debug_dir, body_top=0, overlap_px=0):
    """Save annotated page image, column crops, and detection info."""
    from PIL import Image, ImageDraw, ImageFont

    debug_dir.mkdir(parents=True, exist_ok=True)
    width, height = image.size

    # Annotated full page
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for i, bx in enumerate(boundaries):
        if 0 < bx < width:
            draw.line([(bx, 0), (bx, height)], fill="blue", width=2)
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
    for i in range(len(boundaries) - 1):
        cx = (boundaries[i] + boundaries[i + 1]) // 2
        draw.text((cx - 10, 10), str(i + 1), fill="blue")
    annotated.save(debug_dir / "page_annotated.png")

    # Column crops (with overlap padding matching OCR crops)
    for i in range(len(boundaries) - 1):
        left = max(0, boundaries[i] - overlap_px)
        right = min(width, boundaries[i + 1] + overlap_px)
        if right - left < 30:
            continue
        col_img = image.crop((left, body_top, right, height))
        col_img.save(debug_dir / f"column_{i + 1}_crop.png")

    # Detection info
    info_lines = [
        f"Image size: {width} x {height}",
        f"Body top: {body_top}",
        f"Overlap padding: {overlap_px}px",
        f"Boundaries: {boundaries}",
        f"Columns: {len(boundaries) - 1}",
        "",
    ]
    for i in range(len(boundaries) - 1):
        w = boundaries[i + 1] - boundaries[i]
        pad_left = min(overlap_px, boundaries[i])
        pad_right = min(overlap_px, width - boundaries[i + 1])
        crop_w = w + pad_left + pad_right
        info_lines.append(
            f"  Column {i + 1}: x={boundaries[i]}-{boundaries[i + 1]}, "
            f"width={w}px, crop={crop_w}px (pad L={pad_left} R={pad_right})"
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

    Returns (None, [column_images]).
    Header detection is not performed (headers that don't span full width
    are handled fine by per-column OCR).
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

    # Crop column images, skipping narrow artifacts
    columns = []
    final_boundaries = [boundaries[0]]
    for i in range(len(boundaries) - 1):
        left = boundaries[i]
        right = boundaries[i + 1]
        if right - left < 30:
            continue
        # Apply overlap padding, clamped to image bounds
        crop_left = max(0, left - overlap_px)
        crop_right = min(width, right + overlap_px)
        col_img = image.crop((crop_left, body_top, crop_right, height))
        columns.append(col_img)
        final_boundaries.append(right)

    if debug_dir:
        _save_debug_images(image, final_boundaries, debug_dir, body_top,
                           overlap_px=overlap_px)

    if not columns:
        return (None, [image])

    return (None, columns)
