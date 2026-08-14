from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon


OUTPUT = Path(__file__).with_name("lcp_friction_pyramid.png")


def project(point):
    """Orthographically project contact-frame coordinates (x, y, n)."""
    x, y, normal = np.asarray(point, dtype=float)
    return np.array([x + y, normal + 0.28 * (y - x)])


def add_arrow(ax, start, end, color, width=2.8, mutation_scale=18, zorder=10):
    arrow = FancyArrowPatch(
        project(start),
        project(end),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=width,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def add_polyline(ax, points, **kwargs):
    projected = np.array([project(point) for point in points])
    ax.plot(projected[:, 0], projected[:, 1], **kwargs)


def main():
    background = "#fbfcfe"
    blue = "#0878c9"
    blue_fill = "#86c5ee"
    charcoal = "#515b65"
    projection_gray = "#8c969f"
    red = "#d6291c"
    black = "#161a1e"

    fig, ax = plt.subplots(figsize=(8.6, 5.9), dpi=180)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    cone_height = 3.05
    cone_radius = 1.55
    axis_length = 2.75

    # The four friction directions are exactly the positive and negative
    # tangent-frame axes. The order follows the labels in the figure.
    directions = {
        "d_1": np.array([0.0, -1.0, 0.0]),
        "d_2": np.array([0.0, 1.0, 0.0]),
        "d_3": np.array([1.0, 0.0, 0.0]),
        "d_4": np.array([-1.0, 0.0, 0.0]),
    }

    # Tangent plane. Its edges are generated from the same projected x-y basis
    # used for the axes and the friction directions.
    plane_radius = 2.45
    plane_points = [
        (-plane_radius, -plane_radius, 0.0),
        (plane_radius, -plane_radius, 0.0),
        (plane_radius, plane_radius, 0.0),
        (-plane_radius, plane_radius, 0.0),
    ]
    projected_plane = np.array([project(point) for point in plane_points])
    ax.add_patch(
        Polygon(
            projected_plane,
            closed=True,
            facecolor="#e8ecef",
            edgecolor="none",
            alpha=0.92,
            zorder=0,
        )
    )

    # Tangent-frame axes. Each red d_k arrow below is collinear with one of
    # these two black lines by construction.
    add_polyline(
        ax,
        [(-axis_length, 0.0, 0.0), (axis_length, 0.0, 0.0)],
        color=black,
        linewidth=1.35,
        zorder=2,
    )
    add_polyline(
        ax,
        [(0.0, -axis_length, 0.0), (0.0, axis_length, 0.0)],
        color=black,
        linewidth=1.35,
        zorder=2,
    )

    # Exact Coulomb cone. Every point on the top ellipse is obtained by
    # projecting the circle x^2 + y^2 = cone_radius^2 at n = cone_height.
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    cone_circle_3d = np.column_stack(
        [
            cone_radius * np.cos(theta),
            cone_radius * np.sin(theta),
            np.full_like(theta, cone_height),
        ]
    )
    cone_circle_2d = np.array([project(point) for point in cone_circle_3d])

    left_silhouette = project(
        (-cone_radius / np.sqrt(2.0), -cone_radius / np.sqrt(2.0), cone_height)
    )
    right_silhouette = project(
        (cone_radius / np.sqrt(2.0), cone_radius / np.sqrt(2.0), cone_height)
    )
    origin_2d = project((0.0, 0.0, 0.0))

    ax.add_patch(
        Polygon(
            [origin_2d, left_silhouette, right_silhouette],
            closed=True,
            facecolor=blue_fill,
            edgecolor="none",
            alpha=0.17,
            zorder=3,
        )
    )
    ax.plot(
        [origin_2d[0], left_silhouette[0]],
        [origin_2d[1], left_silhouette[1]],
        color=blue,
        linewidth=2.1,
        zorder=5,
    )
    ax.plot(
        [origin_2d[0], right_silhouette[0]],
        [origin_2d[1], right_silhouette[1]],
        color=blue,
        linewidth=2.1,
        zorder=5,
    )
    ax.fill(
        cone_circle_2d[:, 0],
        cone_circle_2d[:, 1],
        color=blue_fill,
        alpha=0.18,
        zorder=4,
    )
    ax.plot(
        cone_circle_2d[:, 0],
        cone_circle_2d[:, 1],
        color=blue,
        linewidth=2.0,
        zorder=7,
    )

    # Pyramid vertices lie exactly on the circular cone cross-section:
    # x^2 + y^2 = cone_radius^2 and n = cone_height.
    perimeter_labels = ["d_1", "d_3", "d_2", "d_4"]
    top_vertices_3d = {
        label: cone_radius * direction + np.array([0.0, 0.0, cone_height])
        for label, direction in directions.items()
    }
    for point in top_vertices_3d.values():
        assert np.isclose(point[0] ** 2 + point[1] ** 2, cone_radius**2)
        assert np.isclose(point[2], cone_height)

    top_vertices_2d = {
        label: project(point) for label, point in top_vertices_3d.items()
    }

    face_colors = ["#b5bec6", "#c6cdd3", "#a6b0b9", "#bdc5cc"]
    for index, label in enumerate(perimeter_labels):
        next_label = perimeter_labels[(index + 1) % len(perimeter_labels)]
        ax.add_patch(
            Polygon(
                [origin_2d, top_vertices_2d[label], top_vertices_2d[next_label]],
                closed=True,
                facecolor=face_colors[index],
                edgecolor=charcoal,
                linewidth=1.45,
                alpha=0.24,
                zorder=5,
            )
        )

    top_polygon = np.array([top_vertices_2d[label] for label in perimeter_labels])
    ax.add_patch(
        Polygon(
            top_polygon,
            closed=True,
            facecolor="#d8dde1",
            edgecolor=charcoal,
            linewidth=1.55,
            alpha=0.28,
            zorder=6,
        )
    )
    ax.scatter(
        top_polygon[:, 0],
        top_polygon[:, 1],
        s=18,
        facecolor=charcoal,
        edgecolor=background,
        linewidth=0.45,
        zorder=9,
    )

    # Orthogonal projection of the pyramid cross-section onto the tangent
    # plane. The dashed quadrilateral vertices are the endpoints of d_k, and
    # the dotted lines connect every cone vertex to its exact projection.
    projected_vertices_3d = {
        label: cone_radius * direction for label, direction in directions.items()
    }
    projected_vertices_2d = {
        label: project(point) for label, point in projected_vertices_3d.items()
    }
    base_polygon = np.array(
        [projected_vertices_2d[label] for label in perimeter_labels]
    )
    ax.add_patch(
        Polygon(
            base_polygon,
            closed=True,
            facecolor="#c8ced4",
            edgecolor=projection_gray,
            linewidth=1.35,
            linestyle="--",
            alpha=0.18,
            zorder=3,
        )
    )
    for label in perimeter_labels:
        top = top_vertices_2d[label]
        base = projected_vertices_2d[label]
        ax.plot(
            [top[0], base[0]],
            [top[1], base[1]],
            color=projection_gray,
            linewidth=1.0,
            linestyle=(0, (2.5, 3.0)),
            alpha=0.72,
            zorder=4,
        )

    # Contact-frame directions, using the same 3D vectors as the projected
    # pyramid vertices and tangent axes.
    add_arrow(
        ax,
        (0.0, 0.0, 0.03),
        (0.0, 0.0, 2.25),
        red,
        width=3.4,
        mutation_scale=20,
        zorder=10,
    )
    for direction in directions.values():
        add_arrow(
            ax,
            (0.0, 0.0, 0.0),
            cone_radius * direction,
            red,
            width=3.0,
            mutation_scale=18,
            zorder=11,
        )

    ax.scatter(
        [origin_2d[0]],
        [origin_2d[1]],
        s=125,
        facecolor=background,
        edgecolor=black,
        linewidth=1.6,
        zorder=14,
    )

    label_offsets = {
        "d_1": np.array([-0.18, -0.27]),
        "d_2": np.array([0.08, 0.17]),
        "d_3": np.array([0.10, -0.30]),
        "d_4": np.array([-0.32, 0.16]),
    }
    for label, point in projected_vertices_2d.items():
        position = point + label_offsets[label]
        ax.text(
            position[0],
            position[1],
            rf"${label[0]}_{label[-1]}$",
            color=red,
            fontsize=14,
            zorder=15,
        )

    ax.text(0.15, 1.95, r"$\hat{n}$", color=red, fontsize=16, zorder=15)
    ax.text(-0.2, 4.08, "normal", color=black, fontsize=14, ha="right")
    ax.text(2.72, -0.86, "tangent plane", color=black, fontsize=14, ha="left")

    ax.annotate(
        "Coulomb cone",
        xy=right_silhouette + np.array([-0.05, 0.13]),
        xytext=(3.05, 3.85),
        color=blue,
        fontsize=14,
        ha="left",
        arrowprops=dict(
            arrowstyle="-",
            color=blue,
            linewidth=1.6,
            connectionstyle="angle,angleA=180,angleB=45",
        ),
        zorder=15,
    )
    ax.annotate(
        "friction pyramid",
        xy=0.62 * top_vertices_2d["d_3"],
        xytext=(2.65, 2.55),
        color=charcoal,
        fontsize=14,
        ha="left",
        arrowprops=dict(
            arrowstyle="-",
            color=charcoal,
            linewidth=1.4,
            connectionstyle="angle,angleA=180,angleB=70",
        ),
        zorder=15,
    )
    ax.annotate(
        "pyramid projection",
        xy=projected_vertices_2d["d_4"] + np.array([0.12, -0.02]),
        xytext=(-3.25, 1.05),
        color=projection_gray,
        fontsize=12,
        ha="left",
        arrowprops=dict(
            arrowstyle="-",
            color=projection_gray,
            linewidth=1.1,
            connectionstyle="angle,angleA=0,angleB=70",
        ),
        zorder=15,
    )

    ax.set_xlim(-3.7, 4.75)
    ax.set_ylim(-1.35, 4.35)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(
        OUTPUT,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor=background,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
