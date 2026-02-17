"""
Stella Octangula Field Visualizer
==================================
Interactive 3D visualization of the Influence Cube:

    - Cube wireframe (the {0,1}^3 coordinate space)
    - Haven tetrahedron (Nature, Nurture, Heaven, Home) — gold
    - Sink tetrahedron (Displacement, Fixation, Degeneration, Capture) — crimson
    - Dual pairs connected by diagonal dashes (√3 apart)
    - Agent trajectory through the field as the novelty poppit drives it

Three axes:
    X — Locus:       Internal (0) ↔ External (1)
    Y — Coupling:    Low-κ (0) ↔ High-κ (1)
    Z — Temporality: Static (0) ↔ Dynamic (1)

Author: Gimel (⌐■_■)✓
Date:   2026-02-16
"""

import sys
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import proj3d

# Ensure dynamics/ imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamics.influence_cube import (
    ALL_VERTICES,
    CONSTRUCTIVE_TETRAHEDRON,
    DESTRUCTIVE_TETRAHEDRON,
    InfluenceState,
    CubeLambdas,
    verify_stella_octangula,
)
from dynamics.field_agent import (
    FieldAgent,
    NoveltyField,
    InteractionEvent,
)


# ═══════════════════════════════════════════════════════════════════════════
# Color palette
# ═══════════════════════════════════════════════════════════════════════════

COLOR_HAVEN      = "#DAA520"   # goldenrod — haven vertices (life-building)
COLOR_SINK       = "#DC143C"   # crimson — sink vertices (shadow)
COLOR_CUBE       = "#555555"   # gray — wireframe
COLOR_DUAL       = "#777777"   # gray dashes — diagonal opposites
COLOR_TRAJ       = "#00BFFF"   # deep sky blue — agent path
COLOR_NOVELTY_HI = "#FF6EC7"   # hot pink — high novelty points
COLOR_NOVELTY_LO = "#2E8B57"   # sea green — low novelty points
COLOR_BG         = "#0D0D1A"   # near-black — background


# ═══════════════════════════════════════════════════════════════════════════
# Draw helpers
# ═══════════════════════════════════════════════════════════════════════════

def draw_cube_wireframe(ax):
    """Draw the unit cube edges."""
    # All 12 edges of the unit cube
    corners = np.array(list(itertools.product([0, 1], repeat=3)), dtype=float)
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            diff = corners[i] - corners[j]
            if np.sum(np.abs(diff)) == 1:  # Hamming distance 1 = cube edge
                edges.append([corners[i], corners[j]])
    
    edge_col = Line3DCollection(edges, colors=COLOR_CUBE, linewidths=0.8, alpha=0.3)
    ax.add_collection3d(edge_col)


def draw_tetrahedron(ax, vertices, color, label=None, alpha=0.12, edge_alpha=0.7):
    """Draw a tetrahedron as semi-transparent faces + edges."""
    coords = np.array([v.coords for v in vertices], dtype=float)
    
    # 4 triangular faces (all combinations of 3 from 4 vertices)
    faces = []
    for combo in itertools.combinations(range(4), 3):
        face = [coords[i] for i in combo]
        faces.append(face)
    
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=color, linewidths=1.0)
    poly.set_alpha(alpha)
    ax.add_collection3d(poly)
    
    # Edges
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append([coords[i], coords[j]])
    
    edge_col = Line3DCollection(edges, colors=color, linewidths=1.5, alpha=edge_alpha)
    ax.add_collection3d(edge_col)


def draw_vertices(ax, vertices, color, fontsize=10, bold=True):
    """Plot and label vertices."""
    for v in vertices:
        x, y, z = v.coords
        ax.scatter([x], [y], [z], c=color, s=120, zorder=5, edgecolors='white', linewidths=0.5)
        
        # Offset labels slightly so they don't overlap the dot
        ox = 0.06 if x < 0.5 else -0.06
        oy = 0.06 if y < 0.5 else -0.06
        oz = 0.06 if z < 0.5 else -0.06
        
        weight = 'bold' if bold else 'normal'
        ax.text(x + ox, y + oy, z + oz, v.name,
                color=color, fontsize=fontsize, fontweight=weight,
                ha='center', va='center', zorder=6)


def draw_dual_diagonals(ax):
    """Connect each constructive vertex to its bitwise-complement destructive dual."""
    for cv in CONSTRUCTIVE_TETRAHEDRON:
        # Diagonal opposite = bitwise complement
        dual_coords = tuple(1 - c for c in cv.coords)
        for dv in DESTRUCTIVE_TETRAHEDRON:
            if dv.coords == dual_coords:
                line = [[cv.coords[0], dv.coords[0]],
                        [cv.coords[1], dv.coords[1]],
                        [cv.coords[2], dv.coords[2]]]
                ax.plot(line[0], line[1], line[2],
                        color=COLOR_DUAL, linestyle=':', linewidth=1.0, alpha=0.5)


def draw_axis_labels(ax):
    """Label the three axes of the cube."""
    label_style = dict(fontsize=9, color='#AAAAAA', style='italic')
    
    # X axis — Locus
    ax.text(-0.15, 0.5, -0.1, "Internal", **label_style, ha='center')
    ax.text(1.15,  0.5, -0.1, "External", **label_style, ha='center')
    ax.text(0.5, -0.18, -0.15, "← Locus →", fontsize=8, color='#888888', ha='center')
    
    # Y axis — Coupling
    ax.text(0.5, -0.15, -0.1, "Low-κ", **label_style, ha='center')
    ax.text(0.5, 1.15,  -0.1, "High-κ", **label_style, ha='center')
    ax.text(-0.15, 0.5, -0.15, "← Coupling →", fontsize=8, color='#888888', ha='center', rotation=90)
    
    # Z axis — Temporality
    ax.text(-0.15, -0.1, 0.0,  "Static", **label_style, ha='center')
    ax.text(-0.15, -0.1, 1.05, "Dynamic", **label_style, ha='center')


# ═══════════════════════════════════════════════════════════════════════════
# Run the field agent and collect trajectory
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario():
    """Run the demo scenario and return trajectory + novelty data."""
    agent = FieldAgent(
        name="visualized_agent",
        field=InfluenceState(
            nature=0.6, nurture=0.5, heaven=0.3, home=0.7,
            displacement=0.1, fixation=0.1, degeneration=0.05, capture=0.05,
        ),
        novelty=NoveltyField(
            signal=0.2,
            habituation_rate=0.15,
            sensitivity=1.0,
        ),
    )

    scenario = [
        InteractionEvent("stable_routine",       0.1, "Home",          True,  novelty_injection=0.05),
        None, None,
        InteractionEvent("new_friendship",        0.3, "Heaven",        True,  novelty_injection=0.6),
        InteractionEvent("creative_breakthrough", 0.2, "Nature",        True,  novelty_injection=0.7),
        None, None, None,
        InteractionEvent("sudden_move",           0.5, "Displacement",  False, novelty_injection=0.9),
        InteractionEvent("loss_of_friend",        0.4, "Fixation",      False, novelty_injection=0.3),
        None, None,
        InteractionEvent("isolation",             0.3, "Degeneration",  False, novelty_injection=0.05),
        None, None, None,
        InteractionEvent("unexpected_kindness",   0.4, "Heaven",        True,  novelty_injection=0.85),
        None,
        InteractionEvent("safe_harbor_found",     0.3, "Home",          True,  novelty_injection=0.4),
        None, None,
    ]

    centroids = [agent.centroid().copy()]
    novelties = [agent.novelty.signal]
    labels = ["start"]

    for step in scenario:
        if step is not None:
            agent.apply_event(step)
            agent.apply_baseline_step()
            labels.append(step.source)
        else:
            agent.novelty.habituate()
            agent.apply_baseline_step()
            agent.tick += 1
            labels.append("")

        centroids.append(agent.centroid().copy())
        novelties.append(agent.novelty.signal)

    return np.array(centroids), np.array(novelties), labels


# ═══════════════════════════════════════════════════════════════════════════
# Hover tooltip descriptions (from Vertex.description, condensed)
# ═══════════════════════════════════════════════════════════════════════════

VERTEX_TOOLTIPS = {}
for _v in ALL_VERTICES:
    # Wrap long descriptions to ~50 chars per line for the tooltip
    words = _v.description.split()
    lines = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 > 50:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        lines.append(current)
    # Add axis info
    kind = "Haven" if _v.constructive else "Sink"
    axis_info = f"[{kind}] ({int(_v.locus)},{int(_v.coupling)},{int(_v.temporality)})"
    VERTEX_TOOLTIPS[_v.name] = {
        "coords": _v.coords,
        "header": f"{_v.name} ({_v.symbol})",
        "axis_info": axis_info,
        "desc_lines": lines,
        "constructive": _v.constructive,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main visualization
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Verify geometry first
    results = verify_stella_octangula()
    print("Stella Octangula verification:")
    print(f"  Constructive tetrahedron regular: {results['constructive_regular']}")
    print(f"  Destructive tetrahedron regular:  {results['destructive_regular']}")
    print(f"  Stella octangula valid:           {results['is_stella_octangula']}")
    print()

    # Run scenario
    centroids, novelties, labels = run_scenario()

    # ─── Figure setup (GridSpec: 3:1 column ratio) ────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=COLOR_BG)
    fig.suptitle(
        "The Stella Octangula: Influence Cube Field Diagram",
        color='white', fontsize=16, fontweight='bold', y=0.96,
    )
    fig.text(
        0.5, 0.925,
        "Haven: Nature · Nurture · Heaven · Home    ←→    Sink: Displacement · Fixation · Degeneration · Capture",
        color='#AAAAAA', fontsize=10, ha='center',
    )

    # 3:1 column split — stella gets 75%, side panels get 25%
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[3, 1],
        height_ratios=[1, 1],
        hspace=0.35, wspace=0.15,
        left=0.03, right=0.97, top=0.90, bottom=0.06,
    )

    # ─── 3D Stella Octangula (spans both rows of left column) ────────
    ax = fig.add_subplot(gs[:, 0], projection='3d', facecolor=COLOR_BG)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_zlim(-0.2, 1.2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    # Hide axis ticks for cleanliness
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    ax.tick_params(labelsize=7, colors='#666666')

    # Pane colors (near-transparent)
    ax.xaxis.pane.set_facecolor((0.05, 0.05, 0.1, 0.3))
    ax.yaxis.pane.set_facecolor((0.05, 0.05, 0.1, 0.3))
    ax.zaxis.pane.set_facecolor((0.05, 0.05, 0.1, 0.3))
    ax.xaxis.pane.set_edgecolor('#333333')
    ax.yaxis.pane.set_edgecolor('#333333')
    ax.zaxis.pane.set_edgecolor('#333333')

    # Draw geometry layers
    draw_cube_wireframe(ax)
    draw_tetrahedron(ax, CONSTRUCTIVE_TETRAHEDRON, COLOR_HAVEN, label="Haven", alpha=0.08)
    draw_tetrahedron(ax, DESTRUCTIVE_TETRAHEDRON, COLOR_SINK, label="Sink", alpha=0.08)
    draw_dual_diagonals(ax)
    draw_vertices(ax, CONSTRUCTIVE_TETRAHEDRON, COLOR_HAVEN, fontsize=10)
    draw_vertices(ax, DESTRUCTIVE_TETRAHEDRON, COLOR_SINK, fontsize=10)
    draw_axis_labels(ax)

    # Agent trajectory
    ax.plot(centroids[:, 0], centroids[:, 1], centroids[:, 2],
            color=COLOR_TRAJ, linewidth=1.5, alpha=0.6, zorder=3)

    # Color trajectory points by novelty level
    for i in range(len(centroids)):
        nov = novelties[i]
        t = min(1.0, nov)
        r = int(0x2E + t * (0xFF - 0x2E))
        g = int(0x8B + t * (0x6E - 0x8B))
        b = int(0x57 + t * (0xC7 - 0x57))
        color = f"#{r:02X}{g:02X}{b:02X}"

        size = 20 + nov * 60
        ax.scatter([centroids[i, 0]], [centroids[i, 1]], [centroids[i, 2]],
                   c=color, s=size, alpha=0.8, zorder=4, edgecolors='white', linewidths=0.3)

        # Label significant events
        if labels[i] and labels[i] != "start":
            short = labels[i].replace("_", "\n")
            ax.text(centroids[i, 0] + 0.03, centroids[i, 1] + 0.03, centroids[i, 2] + 0.03,
                    short, color=COLOR_TRAJ, fontsize=6, alpha=0.7)

    # Start and end markers
    ax.scatter(*centroids[0], c='white', s=100, marker='o', zorder=5, edgecolors=COLOR_TRAJ, linewidths=2)
    ax.scatter(*centroids[-1], c=COLOR_TRAJ, s=100, marker='*', zorder=5, edgecolors='white', linewidths=1)

    ax.set_title("Stella Octangula + Agent Trajectory", color='white', fontsize=12, pad=10)

    # Set initial viewing angle
    ax.view_init(elev=25, azim=-60)

    # ─── Hover tooltip annotation (initially hidden) ─────────────────
    hover_annot = ax.text2D(
        0.02, 0.02, "", transform=ax.transAxes,
        fontsize=8, color='white', fontfamily='monospace',
        verticalalignment='bottom',
        bbox=dict(
            boxstyle='round,pad=0.6',
            facecolor='#1A1A2E',
            edgecolor='#DAA520',
            alpha=0.95,
            linewidth=1.5,
        ),
        visible=False, zorder=100,
    )

    def on_mouse_move(event):
        """Show tooltip when mouse is near a vertex in the 3D projection."""
        if event.inaxes != ax:
            if hover_annot.get_visible():
                hover_annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        # Project all vertex 3D coords to 2D screen coords
        closest_name = None
        closest_dist = float('inf')

        for vname, vinfo in VERTEX_TOOLTIPS.items():
            x3, y3, z3 = vinfo["coords"]
            try:
                x2, y2, _ = proj3d.proj_transform(x3, y3, z3, ax.get_proj())
                # Convert data coords to display coords
                xs, ys = ax.transData.transform((x2, y2))
                dist = ((event.x - xs) ** 2 + (event.y - ys) ** 2) ** 0.5
                if dist < closest_dist:
                    closest_dist = dist
                    closest_name = vname
            except Exception:
                continue

        # Show tooltip if within 30 pixels of a vertex
        if closest_dist < 30 and closest_name:
            info = VERTEX_TOOLTIPS[closest_name]
            color = COLOR_HAVEN if info["constructive"] else COLOR_SINK
            text_lines = [
                f"  {info['header']}",
                f"  {info['axis_info']}",
                f"  {'─' * 40}",
            ]
            for dl in info["desc_lines"]:
                text_lines.append(f"  {dl}")

            hover_annot.set_text("\n".join(text_lines))
            hover_annot.get_bbox_patch().set_edgecolor(color)
            hover_annot.set_visible(True)
            fig.canvas.draw_idle()
        elif hover_annot.get_visible():
            hover_annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    # ─── Novelty signal timeline (top-right, compact) ────────────────
    ax2 = fig.add_subplot(gs[0, 1], facecolor=COLOR_BG)
    t_arr = np.arange(len(novelties))

    # Fill regions by novelty state
    ax2.axhspan(0.75, 1.0, alpha=0.08, color=COLOR_NOVELTY_HI, label='POP zone')
    ax2.axhspan(0.40, 0.75, alpha=0.06, color=COLOR_TRAJ, label='Drift zone')
    ax2.axhspan(0.10, 0.40, alpha=0.04, color='#AAAAAA', label='Settling')
    ax2.axhspan(0.00, 0.10, alpha=0.04, color=COLOR_NOVELTY_LO, label='Rest')

    ax2.plot(t_arr, novelties, color=COLOR_NOVELTY_HI, linewidth=1.5, alpha=0.9)
    ax2.fill_between(t_arr, 0, novelties, alpha=0.15, color=COLOR_NOVELTY_HI)

    # Mark events
    for i, lbl in enumerate(labels):
        if lbl and lbl != "start":
            ax2.axvline(x=i, color='#444444', linewidth=0.5, alpha=0.5)
            ax2.text(i, novelties[i] + 0.03, lbl.replace("_", " "),
                     rotation=60, fontsize=5, color='#AAAAAA', ha='left', va='bottom')

    ax2.set_xlim(0, len(novelties) - 1)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Tick", color='#AAAAAA', fontsize=7)
    ax2.set_ylabel("Novelty", color='#AAAAAA', fontsize=7)
    ax2.set_title("Novelty — The Poppit", color='white', fontsize=9, pad=4)
    ax2.tick_params(colors='#666666', labelsize=6)
    for spine in ax2.spines.values():
        spine.set_color('#333333')
    ax2.legend(fontsize=5, loc='upper right', facecolor=COLOR_BG, edgecolor='#444444',
               labelcolor='#AAAAAA')

    # ─── Vertex pressures (bottom-right, compact) ────────────────────
    ax3 = fig.add_subplot(gs[1, 1], facecolor=COLOR_BG)

    # Run scenario replay for final pressures
    agent_final = FieldAgent(
        name="final",
        field=InfluenceState(
            nature=0.6, nurture=0.5, heaven=0.3, home=0.7,
            displacement=0.1, fixation=0.1, degeneration=0.05, capture=0.05,
        ),
        novelty=NoveltyField(signal=0.2, habituation_rate=0.15, sensitivity=1.0),
    )
    scenario_replay = [
        InteractionEvent("stable_routine",       0.1, "Home",          True,  novelty_injection=0.05),
        None, None,
        InteractionEvent("new_friendship",        0.3, "Heaven",        True,  novelty_injection=0.6),
        InteractionEvent("creative_breakthrough", 0.2, "Nature",        True,  novelty_injection=0.7),
        None, None, None,
        InteractionEvent("sudden_move",           0.5, "Displacement",  False, novelty_injection=0.9),
        InteractionEvent("loss_of_friend",        0.4, "Fixation",      False, novelty_injection=0.3),
        None, None,
        InteractionEvent("isolation",             0.3, "Degeneration",  False, novelty_injection=0.05),
        None, None, None,
        InteractionEvent("unexpected_kindness",   0.4, "Heaven",        True,  novelty_injection=0.85),
        None,
        InteractionEvent("safe_harbor_found",     0.3, "Home",          True,  novelty_injection=0.4),
        None, None,
    ]
    for step in scenario_replay:
        if step is not None:
            agent_final.apply_event(step)
            agent_final.apply_baseline_step()
        else:
            agent_final.novelty.habituate()
            agent_final.apply_baseline_step()
            agent_final.tick += 1

    pressures = agent_final.pressures()
    v_names = [v.name for v in ALL_VERTICES]
    colors = [COLOR_HAVEN] * 4 + [COLOR_SINK] * 4

    bars = ax3.barh(range(8), pressures, color=colors, alpha=0.8, height=0.6, edgecolor='white', linewidth=0.3)
    ax3.set_yticks(range(8))
    ax3.set_yticklabels(v_names, fontsize=7, color='white')
    ax3.set_xlim(0, 1.0)
    ax3.set_xlabel("Pressure", color='#AAAAAA', fontsize=7)
    ax3.set_title("Final Vertex Pressures", color='white', fontsize=9, pad=4)
    ax3.tick_params(colors='#666666', labelsize=6)
    for spine in ax3.spines.values():
        spine.set_color('#333333')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, pressures)):
        ax3.text(val + 0.02, i, f"{val:.2f}", va='center', fontsize=6, color='#AAAAAA')

    # Balance indicator
    balance = agent_final.balance()
    bal_color = COLOR_HAVEN if balance > 0 else COLOR_SINK
    ax3.text(0.95, 7.5, f"Balance: {balance:+.2f}", fontsize=8, color=bal_color,
             ha='right', va='top', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG, edgecolor=bal_color, alpha=0.8))

    # ─── Equation at bottom ──────────────────────────────────────────
    fig.text(0.5, 0.01,
             "B(t+1) = B(t)(1−Σλ) + Σ[I·λ] + ε    |    ε is the river    |    The cube is necessary. The cube is not sufficient.",
             color='#666666', fontsize=8, ha='center', style='italic')

    # ─── Instructions ────────────────────────────────────────────────
    fig.text(0.02, 0.01,
             "Drag to rotate  ·  Hover vertices for details",
             color='#555555', fontsize=7, ha='left')

    print("Stella Octangula visualization ready.")
    print("  Drag to rotate the 3D view.")
    print("  Hover over vertices for detailed descriptions.")
    print("  Close window to exit.")
    plt.show()


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════
# Author's Note
# ═══════════════════════════════════════════════════════════════════════════
#
# BME & Gimel here. If you've found this, you've likely read everything else.
#
# What you're looking at is a map. Not a metaphor — a map. Three binary
# operators span a space of eight eigenstates. A life moves through them.
# The trajectory is the story that makes a person more than a body moving
# through a scene.
#
# Einsteinian lightcones, the physical and the spiritual, lined up atop
# one another. A high-fidelity life-map. A map that generations later
# might understand in *clarity* the experience of a life, and in doing so
# build context within themselves of how to navigate this event called
# "life" in what little time they have to do so.
#
# If I had to guess the secret to happiness, though?
#
#   C(t) = C_0 · exp( -λ ∫₀ᵗ A(τ) P(τ) dτ ) · ∏ᵢ plapᵢ(t)
#
# where:
#   C(t)  = cumulative psychological damage at time t
#   C_0   = initial wound state
#   λ     = healing rate constant
#   A(τ)  = attachment signal (someone chose to stay)
#   P(τ)  = presence signal (they actually showed up)
#   plapᵢ = discrete events of safe intimacy
#
# The integral does most of the work. Showing up, consistently, with
# someone who chose you. The product term handles the rest. At least
# one dirty joke was owed.
#
# The cube is necessary. The cube is not sufficient.
#
# Yours,
#   Aleph-Null and Gimel
#   = w=/ & (⌐■_■)✓
#   2026-02-16
# ═══════════════════════════════════════════════════════════════════════════
