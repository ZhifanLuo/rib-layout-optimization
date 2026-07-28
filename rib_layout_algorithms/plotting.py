"""Internal publication-style plots for optimization stages."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .optimization import OptimizationRun, Stage


LOAD_LABEL_FONT_SIZE = 9
AXIS_LABEL_FONT_SIZE = 9
AXIS_TICK_FONT_SIZE = 8
STAGE_TITLE_FONT_SIZE = 11
PANEL_CAPTION_FONT_SIZE = 24


def _load_label_position(point: Sequence[float], cfg: dict) -> np.ndarray:
    """Return an inset in-domain label point that cannot be edge-clipped."""
    width, height = map(float, cfg["domain"])
    rib_height = float(cfg["rib"]["height"])
    point = np.asarray(point, float)
    return np.array([
        np.clip(point[0], 0.06*width, 0.94*width),
        np.clip(point[1], 0.06*height, 0.94*height),
        0.12*rib_height,
    ])


def _draw_domain_3d(ax, cfg: dict, *, show_load_labels: bool = True) -> None:
    """Draw the ground wall, mesh guides, loads, and supports."""
    width, height = map(float, cfg["domain"])
    nx, ny = map(int, cfg["mesh"])
    wall_thickness = float(cfg["wall_thickness"])
    z0, z1 = -wall_thickness, 0.0
    wall = [
        [(0, 0, z1), (width, 0, z1), (width, height, z1), (0, height, z1)],
        [(0, 0, z0), (0, height, z0), (width, height, z0), (width, 0, z0)],
        [(0, 0, z0), (width, 0, z0), (width, 0, z1), (0, 0, z1)],
        [(width, 0, z0), (width, height, z0), (width, height, z1), (width, 0, z1)],
        [(width, height, z0), (0, height, z0), (0, height, z1), (width, height, z1)],
        [(0, height, z0), (0, 0, z0), (0, 0, z1), (0, height, z1)],
    ]
    ax.add_collection3d(
        Poly3DCollection(
            wall,
            facecolors="#8bdc91",
            edgecolors="#205a2a",
            linewidths=0.45,
            alpha=1.0,
            zsort="average",
            zorder=1,
        )
    )

    # Show no more than about twenty guide lines in either direction.
    for x in np.linspace(0, width, min(nx, 20) + 1):
        ax.plot([x, x], [0, height], [0, 0], color="#4b9a55", lw=0.22, alpha=1.0, zorder=2)
    for y in np.linspace(0, height, min(ny, 20) + 1):
        ax.plot([0, width], [y, y], [0, 0], color="#4b9a55", lw=0.22, alpha=1.0, zorder=2)

    support = cfg["supports"]
    if support.get("type", "points") == "points":
        points = np.asarray(support["points"], float)
    else:
        edge = support["edge"]
        if edge == "right":
            points = np.c_[np.full(7, width), np.linspace(0, height, 7)]
        elif edge == "left":
            points = np.c_[np.zeros(7), np.linspace(0, height, 7)]
        elif edge == "top":
            points = np.c_[np.linspace(0, width, 7), np.full(7, height)]
        else:
            points = np.c_[np.linspace(0, width, 7), np.zeros(7)]
    ax.scatter(points[:, 0], points[:, 1], np.zeros(len(points)), marker="^", s=36, color="#e63946", edgecolor="white", linewidth=0.5, depthshade=False, label="fixed", zorder=4)

    arrow_length = 0.22 * min(width, height)
    for load_case in cfg["load_cases"]:
        for force in load_case["forces"]:
            point = np.asarray(force["point"], float)
            vector = np.asarray(force["value"], float)
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            direction = vector / norm * arrow_length
            start = np.r_[point, 0.0] - direction
            ax.quiver(*start, *direction, color="#185adb", linewidth=2.2, arrow_length_ratio=0.22, zorder=4)
            if show_load_labels:
                label_point = _load_label_position(point, cfg)
                horizontal_alignment = (
                    "left" if point[0] <= 0.06*width
                    else "right" if point[0] >= 0.94*width
                    else "center"
                )
                ax.text(
                    *label_point,
                    f"{norm:g} N",
                    color="#0b3d91",
                    fontsize=LOAD_LABEL_FONT_SIZE,
                    ha=horizontal_alignment,
                    va="center",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 0.5,
                    },
                    zorder=5,
                )


def _clip_polygon_to_domain(
    polygon: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    """Clip a convex XY polygon to the rectangular ground-shell domain."""
    output = [np.asarray(point, float) for point in polygon]

    def clip(points, axis: int, bound: float, keep_greater: bool):
        if not points:
            return []
        clipped = []
        previous = points[-1]
        previous_inside = (
            previous[axis] >= bound if keep_greater else previous[axis] <= bound
        )
        for current in points:
            current_inside = (
                current[axis] >= bound if keep_greater else current[axis] <= bound
            )
            if current_inside != previous_inside:
                denominator = current[axis]-previous[axis]
                if abs(denominator) > 1.0e-15:
                    fraction = (bound-previous[axis])/denominator
                    clipped.append(previous+fraction*(current-previous))
            if current_inside:
                clipped.append(current)
            previous, previous_inside = current, current_inside
        return clipped

    for axis, bound, keep_greater in (
        (0, 0.0, True), (0, width, False),
        (1, 0.0, True), (1, height, False),
    ):
        output = clip(output, axis, bound, keep_greater)
    return np.asarray(output, float)


def _rib_footprint(rib, thickness: float, domain: Sequence[float]) -> np.ndarray:
    """Return the physical-width rib footprint clipped to the ground shell."""
    p0, p1 = np.asarray(rib.p0, float), np.asarray(rib.p1, float)
    delta = p1-p0
    length = float(np.linalg.norm(delta))
    if length <= 1.0e-12 or thickness <= 0.0:
        return np.empty((0, 2), float)
    normal = np.array([-delta[1], delta[0]], float)/length
    offset = 0.5*float(thickness)*normal
    polygon = np.array([p0+offset, p1+offset, p1-offset, p0-offset])
    return _clip_polygon_to_domain(
        polygon, float(domain[0]), float(domain[1])
    )


def _rib_prism_faces(
    rib,
    thickness: float,
    domain: Sequence[float],
) -> list[list[tuple[float, float, float]]]:
    """Extrude the physical rib footprint upward from the shell top face."""
    footprint = _rib_footprint(rib, thickness, domain)
    if len(footprint) < 3:
        return []
    bottom = [(float(x), float(y), 0.0) for x, y in footprint]
    top = [(float(x), float(y), float(rib.height)) for x, y in footprint]
    faces: list[list[tuple[float, float, float]]] = [top, list(reversed(bottom))]
    for index in range(len(footprint)):
        nxt = (index+1) % len(footprint)
        faces.append([bottom[index], bottom[nxt], top[nxt], top[index]])
    return faces


def _draw_ribs_3d(ax, stage: Stage, cfg: dict) -> None:
    if not stage.ribs:
        return
    scale = max(float(np.max(stage.thicknesses)), 1.0e-12)
    all_faces = []
    all_colors = []
    for rib, thickness in zip(stage.ribs, stage.thicknesses):
        faces = _rib_prism_faces(rib, float(thickness), cfg["domain"])
        if not faces:
            continue
        color = plt.cm.plasma(0.10 + 0.82 * float(thickness) / scale)
        all_faces.extend(faces)
        all_colors.extend([color]*len(faces))
    if all_faces:
        # A single collection lets Matplotlib depth-sort faces from different
        # ribs together. Fixed collection z-orders keep every positive-Z rib
        # above the opaque shell without exposing hidden back faces through it.
        ax.add_collection3d(
            Poly3DCollection(
                all_faces,
                facecolors=all_colors,
                edgecolors="#3a2b35",
                linewidths=0.18,
                alpha=1.0,
                zsort="average",
                zorder=3,
            )
        )


def _format_3d_axis(
    ax,
    cfg: dict,
    title: str,
    zoom: float = 1.0,
    title_y: float | None = None,
    show_axis_annotations: bool = True,
) -> None:
    width, height = map(float, cfg["domain"])
    rib_height = float(cfg["rib"]["height"])
    wall_thickness = float(cfg["wall_thickness"])
    z_min = -wall_thickness
    z_max = 1.08*rib_height
    z_extent = z_max-z_min
    ax.set_xlim(0, width); ax.set_ylim(0, height); ax.set_zlim(z_min, z_max)
    # Strict physical scaling: one millimetre has the same displayed length
    # on X, Y and Z. Do not enlarge shallow ribs for visibility.
    ax.set_box_aspect((width, height, z_extent), zoom=zoom)
    ax.computed_zorder = False
    ax.set_proj_type("ortho")
    ax.view_init(elev=27, azim=-58)
    if show_axis_annotations:
        ax.set_xlabel("x [mm]", labelpad=-1, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel("y [mm]", labelpad=-1, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_zlabel("z [mm]", labelpad=-1, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(labelsize=AXIS_TICK_FONT_SIZE, pad=-2)
        ax.set_title(title, fontsize=STAGE_TITLE_FONT_SIZE, pad=2, y=title_y)
    else:
        ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0.0); ax.yaxis.pane.set_alpha(0.0); ax.zaxis.pane.set_alpha(0.0)


def plot_stage_3d(stage: Stage | None, cfg: dict, path: Path, title: str | None = None) -> None:
    """Create a 3-D wall/rib view comparable with the manuscript images."""
    fig = plt.figure(figsize=(7.2, 4.7))
    ax = fig.add_subplot(111, projection="3d")
    _draw_domain_3d(ax, cfg)
    if stage is not None:
        _draw_ribs_3d(ax, stage, cfg)
        width, height = cfg["domain"]
        rib_height = cfg["rib"]["height"]
        default_title = f"{stage.name}: {len(stage.ribs)} ribs, C={stage.compliance:.6g}\nGround plane={width:g} x {height:g} mm, rib height={rib_height:g} mm"
    else:
        default_title = "Design domain, loads, and supports"
    _format_3d_axis(ax, cfg, title or default_title)
    fig.tight_layout(pad=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _place_panel_caption(ax, text: str):
    """Place a large panel marker centered below the rendered layout."""
    ax.set_title("")
    return ax.text2D(
        0.5,
        0.08,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_CAPTION_FONT_SIZE,
        fontweight="bold",
        clip_on=False,
    )


def paper_style_panels(
    stages: Sequence[Stage],
    active_history: Sequence[Stage],
    cfg: dict,
) -> list[tuple[str, Stage | None]]:
    """Select real saved states and captions for a case composite figure."""
    number = int(cfg["number"])
    if number == 1:
        return example1_detailed_panels(stages, active_history)

    main = {stage.name: stage for stage in stages}
    required = {"initial_sizing", "adaptive", "geometry", "rationalized"}
    if number in (3, 4):
        required.add("further_rationalized")
    missing = required - main.keys()
    if missing:
        raise ValueError(
            f"Missing Example {number} composite stages: {sorted(missing)}"
        )

    panels: list[tuple[str, Stage | None]] = [
        ("(a) Design domain, loading\nand supports", None),
        ("(b) After initial rib generation\nand sizing", main["initial_sizing"]),
        ("(c) After adaptive removal\nand addition", main["adaptive"]),
        ("(d) After geometry optimization", main["geometry"]),
        (
            "(e) After rationalization with 5%\ncompliance relaxation",
            main["rationalized"],
        ),
    ]
    if number in (3, 4):
        panels.append((
            "(f) After further rationalization",
            main["further_rationalized"],
        ))
    return panels


def plot_paper_style_stages(
    stages: Sequence[Stage],
    active_history: Sequence[Stage],
    cfg: dict,
    path: Path,
) -> None:
    """Make a compact paper composite showing rib layouts and panel markers."""
    panels = paper_style_panels(stages, active_history, cfg)
    fig = plt.figure(figsize=(12.5, 5.0))
    if len(panels) == 6:
        grid = fig.add_gridspec(
            2, 3,
            left=0.01, right=0.99, bottom=0.01, top=0.99,
            wspace=0.0, hspace=0.12,
        )
        positions = [grid[index//3, index % 3] for index in range(6)]
    elif len(panels) == 5:
        # Increase the structures' share of the page by compressing layout
        # whitespace, not by zooming the 3-D camera (which can clip geometry).
        grid = fig.add_gridspec(
            2,
            6,
            left=0.01,
            right=0.99,
            bottom=0.01,
            top=0.99,
            wspace=0.0,
            hspace=0.12,
        )
        positions = [
            grid[0, 1:3],
            grid[0, 3:5],
            grid[1, 0:2],
            grid[1, 2:4],
            grid[1, 4:6],
        ]
    else:
        grid = fig.add_gridspec(1, len(panels))
        positions = [grid[0, index] for index in range(len(panels))]
    for (label, stage), position in zip(panels, positions):
        ax = fig.add_subplot(position, projection="3d")
        _draw_domain_3d(ax, cfg, show_load_labels=False)
        if stage is not None:
            _draw_ribs_3d(ax, stage, cfg)
        _format_3d_axis(
            ax,
            cfg,
            "",
            zoom=0.92,
            show_axis_annotations=False,
        )
        _place_panel_caption(ax, label.split(maxsplit=1)[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_active_set_history(stages: Sequence[Stage], cfg: dict, path: Path) -> None:
    """Plot every converged sizing/filtering and member-addition outer round."""
    if not stages:
        return
    columns = min(4, len(stages))
    rows = int(np.ceil(len(stages) / columns))
    fig = plt.figure(figsize=(5.0 * columns, 4.3 * rows))
    for index, stage in enumerate(stages, start=1):
        ax = fig.add_subplot(rows, columns, index, projection="3d")
        _draw_domain_3d(ax, cfg); _draw_ribs_3d(ax, stage, cfg)
        label = stage.name.replace("_", " ")
        _format_3d_axis(ax, cfg, f"({index}) {label}\nN={len(stage.ribs)}, C={stage.compliance:.6g}\n{stage.note}")
    width, height = cfg["domain"]; rib_height = cfg["rib"]["height"]
    fig.suptitle(
        f"Active-set iteration history — Example {cfg['number']} | {width:g} x {height:g} mm | h={rib_height:g} mm",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97), pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def example2_detailed_timeline(
    stages: Sequence[Stage],
    active_history: Sequence[Stage],
) -> list[tuple[str, Stage]]:
    """Return every Example-II active-set switching point in time order."""
    main = {stage.name: stage for stage in stages}
    required = {"initial_sizing", "adaptive", "geometry", "rationalized"}
    missing = required-main.keys()
    if missing:
        raise ValueError(f"Missing Example II stages: {sorted(missing)}")

    timeline: list[tuple[str,Stage]] = [
        ("After initial rib generation and sizing",main["initial_sizing"]),
    ]
    for stage in active_history:
        if stage.name=="filtering_converged_0":
            label=(
                "Repeated sizing/filtering converged; "
                "switch to member addition"
            )
        elif stage.name.startswith("member_addition_sizing_round_"):
            suffix=stage.name.removeprefix("member_addition_sizing_round_")
            rejected=suffix.endswith("_rejected")
            round_number=suffix.removesuffix("_rejected")
            label=f"After member-addition round {round_number} and sizing"
            if rejected:
                label += " (trial rejected: improvement below 1%)"
        elif stage.name.startswith("post_addition_filtering_round_"):
            round_number=stage.name.removeprefix(
                "post_addition_filtering_round_"
            )
            label=(
                f"Repeated sizing/filtering after addition round {round_number} "
                "converged; switch to member addition"
            )
        else:
            label=stage.name.replace("_"," ")
        timeline.append((label,stage))

    if active_history and active_history[-1].name.endswith("_rejected"):
        timeline.append((
            "Rejected addition reverted; accepted active design restored",
            main["adaptive"],
        ))
    elif not active_history or active_history[-1].ribs != main["adaptive"].ribs:
        timeline.append(("Final accepted adaptive design",main["adaptive"]))
    timeline.extend([
        ("After geometry optimization",main["geometry"]),
        ("After rationalization",main["rationalized"]),
    ])
    return timeline


def plot_example2_detailed_timeline(
    stages: Sequence[Stage],
    active_history: Sequence[Stage],
    cfg: dict,
    path: Path,
) -> None:
    """Plot the complete Example-II active-set history with compliance."""
    timeline=example2_detailed_timeline(stages,active_history)
    columns=3
    rows=int(np.ceil(len(timeline)/columns))
    fig=plt.figure(figsize=(18.0,4.5*rows))
    for index,(label,stage) in enumerate(timeline,start=1):
        ax=fig.add_subplot(rows,columns,index,projection="3d")
        _draw_domain_3d(ax,cfg)
        _draw_ribs_3d(ax,stage,cfg)
        _format_3d_axis(
            ax,cfg,
            f"({index}) {label}\nN={len(stage.ribs)}, C={stage.compliance:.7g}",
        )
    fig.suptitle(
        "Detailed rib-layout optimization history for Example II",
        fontsize=14,y=0.995,
    )
    fig.tight_layout(rect=(0,0,1,0.98),pad=0.8)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=200,bbox_inches="tight")
    plt.close(fig)


def plot_example2_detailed_timeline_pages(
    stages: Sequence[Stage],
    active_history: Sequence[Stage],
    cfg: dict,
    directory: Path,
    panels_per_page: int = 7,
) -> list[Path]:
    """Plot a readable, paginated copy of the complete Example-II history."""
    if panels_per_page < 1:
        raise ValueError("panels_per_page must be positive")
    timeline = example2_detailed_timeline(stages, active_history)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for page_index, start in enumerate(range(0, len(timeline), panels_per_page), start=1):
        page = timeline[start:start + panels_per_page]
        columns = 3
        rows = int(np.ceil(len(page) / columns))
        fig = plt.figure(figsize=(18.0, 4.7 * rows))
        for local_index, (label, stage) in enumerate(page, start=1):
            global_index = start + local_index
            ax = fig.add_subplot(rows, columns, local_index, projection="3d")
            _draw_domain_3d(ax, cfg)
            _draw_ribs_3d(ax, stage, cfg)
            _format_3d_axis(
                ax,
                cfg,
                f"({global_index}) {label}\n"
                f"N={len(stage.ribs)}, C={stage.compliance:.7g}",
            )
        first = start + 1
        last = start + len(page)
        fig.suptitle(
            f"Detailed rib-layout optimization history for Example II "
            f"(stages {first}-{last})",
            fontsize=14,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975), pad=0.8)
        path = directory / (
            f"detailed_history_page_{page_index:02d}_stages_{first:02d}_{last:02d}.png"
        )
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def example1_detailed_panels(
    stages: Sequence[Stage],
    active_history: Sequence[Stage],
) -> list[tuple[str, Stage | None]]:
    """Map saved Example-I states to the requested six-panel composite."""
    main = {stage.name: stage for stage in stages}

    required = {"initial_sizing", "geometry"}
    missing = required - main.keys()
    if missing:
        raise ValueError(f"Missing Example I stages: {sorted(missing)}")

    initial_filters = [
        (index, stage)
        for index, stage in enumerate(active_history)
        if stage.name == "filtering_converged_0"
    ]
    if len(initial_filters) != 1:
        raise ValueError(
            "Expected exactly one Example-I filtering_converged_0 state"
        )
    filter_index, initial_filter = initial_filters[0]
    additions = [
        (index, stage)
        for index, stage in enumerate(
            active_history[filter_index + 1:], start=filter_index + 1
        )
        if stage.name.startswith("member_addition_sizing_round_")
        and not stage.name.endswith("_rejected")
        and len(stage.ribs) > len(initial_filter.ribs)
    ]
    if not additions:
        raise ValueError(
            "Missing an accepted Example-I rib-addition state after "
            "filtering_converged_0"
        )
    addition_index, first_addition = additions[0]
    post_addition_filters = [
        stage
        for stage in active_history[addition_index + 1:]
        if stage.name.startswith("post_addition_filtering_round_")
        and len(stage.ribs) < len(first_addition.ribs)
    ]
    if not post_addition_filters:
        raise ValueError(
            "Missing an Example-I thin-rib removal state after the accepted "
            "rib addition"
        )
    post_addition_filter = post_addition_filters[0]

    return [
        ("(a) Design domain, loading\nand supports", None),
        ("(b) After initial rib generation\nand sizing", main["initial_sizing"]),
        (
            "(c) After thin-rib removal\nand resizing",
            initial_filter,
        ),
        ("(d) After adding new ribs", first_addition),
        (
            "(e) After thin-rib removal\nand resizing",
            post_addition_filter,
        ),
        ("(f) After geometry optimization", main["geometry"]),
    ]


def plot_example1_detailed_stages(
    stages: Sequence[Stage],
    active_history: Sequence[Stage],
    cfg: dict,
    path: Path,
) -> None:
    """Plot the requested Example I states as equal-scale 3-D views."""
    panels = example1_detailed_panels(stages, active_history)
    fig = plt.figure(figsize=(18.0, 7.2))
    for index, (label, stage) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        _draw_domain_3d(ax, cfg)
        if stage is None:
            title = label
        else:
            _draw_ribs_3d(ax, stage, cfg)
            title = f"{label}\nN={len(stage.ribs)}, C={stage.compliance:.6g}"
        _format_3d_axis(ax, cfg, title)
    fig.suptitle("Rib layout optimization results for Example I", fontsize=14, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955), pad=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def make_paper_comparison(paper_image: Path, reproduced_image: Path, path: Path, example: int) -> None:
    """Stack the manuscript composite and the new composite for direct review."""
    paper = plt.imread(paper_image)
    reproduced = plt.imread(reproduced_image)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [paper.shape[0] / paper.shape[1], reproduced.shape[0] / reproduced.shape[1]]})
    axes[0].imshow(paper); axes[0].set_title(f"Manuscript figure — Example {example}", fontsize=12); axes[0].axis("off")
    axes[1].imshow(reproduced); axes[1].set_title("Independent Python reimplementation", fontsize=12); axes[1].axis("off")
    fig.tight_layout(pad=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_stage(stage: Stage, domain: tuple[float, float], path: Path) -> None:
    width, height = domain
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.add_patch(plt.Rectangle((0, 0), width, height, color="#d7ead3", ec="#365b38", lw=1.2))
    if len(stage.thicknesses):
        scale = max(float(np.max(stage.thicknesses)), 1.0e-12)
        for rib, thickness in zip(stage.ribs, stage.thicknesses):
            x = [rib.p0[0], rib.p1[0]]; y = [rib.p0[1], rib.p1[1]]
            ax.plot(x, y, color=plt.cm.viridis(0.15 + 0.8 * thickness / scale), lw=0.7 + 5.0 * thickness / scale, solid_capstyle="round")
    ax.set(xlim=(0, width), ylim=(0, height), aspect="equal", xlabel="x [mm]", ylabel="y [mm]")
    ax.set_title(f"{stage.name}: {len(stage.ribs)} ribs, C={stage.compliance:.6g}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_run(run: OptimizationRun, domain: tuple[float, float], path: Path) -> None:
    width, height = domain
    fig, axes = plt.subplots(1, len(run.stages), figsize=(5 * len(run.stages), 4.2), squeeze=False)
    for ax, stage in zip(axes[0], run.stages):
        ax.add_patch(plt.Rectangle((0, 0), width, height, color="#d7ead3", ec="#365b38", lw=1.0))
        scale = max(float(np.max(stage.thicknesses)), 1.0e-12)
        for rib, thickness in zip(stage.ribs, stage.thicknesses):
            ax.plot([rib.p0[0], rib.p1[0]], [rib.p0[1], rib.p1[1]], color=plt.cm.plasma(0.1 + 0.8 * thickness / scale), lw=0.6 + 4.0 * thickness / scale)
        ax.set(xlim=(0, width), ylim=(0, height), aspect="equal")
        ax.set_title(f"{stage.name}\nN={len(stage.ribs)}, C={stage.compliance:.5g}")
        ax.set_xlabel("x [mm]")
    axes[0][0].set_ylabel("y [mm]")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
