"""
Generisanje GIF animacije sistema masa-opruga-prigušivač.

Pokreni sa: python animacija.py
Rezultat: animacija.gif
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from simulacija import A, P, t, resi

# === Rešavanje jednačine ===
rezultat = resi()

# === Parametri animacije ===
SCALE = 1500
STEP = 100
FPS = 30
N_COILS = 10

# Pod-odabrani podaci
idx = np.arange(0, len(t), STEP)
t_sub = t[idx]
x_sub = rezultat[idx, 0]
v_sub = rezultat[idx, 1]
F_sub = A * P * np.sin(2 * np.pi * t_sub)

# Geometrija šeme
WALL_X = 0.0
REST_X = 4.0
MASS_W = 1.0
MASS_H = 1.2
SPRING_Y = 0.8
DAMP_Y = -0.4
DAMP_CYL_W = 1.0
DAMP_CYL_H = 0.5


def spring_xy(x_start, x_end, y, n_coils=N_COILS, amp=0.25):
    """Generiše x,y koordinate zig-zag opruge."""
    lead = 0.3
    pts = 2 * n_coils + 1
    x_lead_in = np.array([x_start, x_start + lead])
    x_lead_out = np.array([x_end - lead, x_end])
    coil_x = np.linspace(x_start + lead, x_end - lead, pts)
    coil_y = np.array([amp * ((-1) ** i) for i in range(pts)])
    xs = np.concatenate([x_lead_in, coil_x, x_lead_out])
    ys = np.concatenate([[0, 0], coil_y, [0, 0]])
    return xs, ys + y


def create_animation():
    """Kreira animaciju i čuva kao GIF."""
    fig, (ax_sch, ax_plot) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={'height_ratios': [1, 1.2]}
    )
    fig.subplots_adjust(hspace=0.30)

    # === Gornji panel — šema ===
    ax_sch.set_xlim(-1, 9)
    ax_sch.set_ylim(-1.8, 2.0)
    ax_sch.set_aspect('equal')
    ax_sch.axis('off')
    ax_sch.set_title('Šema sistema', fontsize=13)

    # Zid
    wall = plt.Rectangle((-0.6, -1.5), 0.6, 3.2, fc='lightgray', ec='k', hatch='///')
    ax_sch.add_patch(wall)
    ax_sch.plot([0, 0], [-1.5, 1.7], 'k', lw=2)

    # Opruga
    sx, sy = spring_xy(WALL_X, REST_X, SPRING_Y)
    spring_line, = ax_sch.plot(sx, sy, 'k', lw=1.5)

    # Prigušivač
    rod_left, = ax_sch.plot([], [], 'k', lw=2)
    cyl_left, = ax_sch.plot([], [], 'k', lw=2)
    cyl_top, = ax_sch.plot([], [], 'k', lw=2)
    cyl_bot, = ax_sch.plot([], [], 'k', lw=2)
    piston, = ax_sch.plot([], [], 'k', lw=3)
    rod_right, = ax_sch.plot([], [], 'k', lw=2)

    # Masa
    mass_patch = FancyBboxPatch(
        (REST_X, -MASS_H / 2), MASS_W, MASS_H,
        boxstyle="round,pad=0.05", fc='#4a90d9', ec='k', lw=2, zorder=5
    )
    ax_sch.add_patch(mass_patch)
    mass_label = ax_sch.text(
        REST_X + MASS_W / 2, 0, 'M', ha='center', va='center',
        fontsize=16, fontweight='bold', color='white', zorder=6
    )

    # Sila
    force_arrow = FancyArrowPatch(
        (REST_X + MASS_W, 0), (REST_X + MASS_W + 1.5, 0),
        arrowstyle='->', mutation_scale=20, color='red', lw=2.5, zorder=5
    )
    ax_sch.add_patch(force_arrow)
    force_text = ax_sch.text(
        REST_X + MASS_W + 1.6, 0.35, '', fontsize=10, color='red',
        fontweight='bold', zorder=6
    )

    # Vreme
    time_text = ax_sch.text(
        7.5, 1.6, '', fontsize=11, ha='right',
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.8)
    )

    # Labele
    ax_sch.text((WALL_X + REST_X) / 2, SPRING_Y + 0.55, 'opruga (c)',
                ha='center', fontsize=9, style='italic', color='gray')
    ax_sch.text((WALL_X + REST_X) / 2, DAMP_Y - 0.55, 'prigušivač (μ)',
                ha='center', fontsize=9, style='italic', color='gray')

    # === Donji panel — x(t) i v(t) ===
    ax_plot.set_xlim(t_sub[0], t_sub[-1])
    x_max = np.max(np.abs(x_sub)) * 1.15
    v_max = np.max(np.abs(v_sub)) * 1.15
    ax_plot.set_ylim(-max(x_max, v_max), max(x_max, v_max))
    ax_plot.set_xlabel('t [s]')
    ax_plot.set_title('Vremenski odziv', fontsize=13)
    ax_plot.grid(True, alpha=0.3)

    line_x, = ax_plot.plot([], [], 'b', lw=1.5, label='x(t)')
    line_v, = ax_plot.plot([], [], 'g', lw=1.0, alpha=0.7, label='v(t)')
    cursor, = ax_plot.plot([], [], 'r-', lw=0.8, alpha=0.6)
    ax_plot.legend(loc='upper right')

    def update(i):
        xi = x_sub[i]
        fi = F_sub[i]
        ti = t_sub[i]
        mass_x = REST_X + xi * SCALE

        # Opruga
        sx, sy = spring_xy(WALL_X, mass_x, SPRING_Y)
        spring_line.set_data(sx, sy)

        # Prigušivač
        cyl_left_x = (WALL_X + mass_x) / 2 - DAMP_CYL_W / 2
        cyl_right_x = cyl_left_x + DAMP_CYL_W
        dy = DAMP_Y
        hh = DAMP_CYL_H / 2

        rod_left.set_data([WALL_X, cyl_left_x], [dy, dy])
        cyl_left.set_data([cyl_left_x, cyl_left_x], [dy - hh, dy + hh])
        cyl_top.set_data([cyl_left_x, cyl_right_x], [dy + hh, dy + hh])
        cyl_bot.set_data([cyl_left_x, cyl_right_x], [dy - hh, dy - hh])
        piston_x = cyl_left_x + DAMP_CYL_W * 0.5 + (mass_x - REST_X) * 0.3
        piston_x = np.clip(piston_x, cyl_left_x + 0.05, cyl_right_x - 0.05)
        piston.set_data([piston_x, piston_x], [dy - hh + 0.03, dy + hh - 0.03])
        rod_right.set_data([piston_x, mass_x], [dy, dy])

        # Masa
        mass_patch.set_x(mass_x)
        mass_label.set_x(mass_x + MASS_W / 2)

        # Sila
        f_scale = fi / (A * P) * 2.0
        if fi != 0:
            arrow_end = mass_x + MASS_W + abs(f_scale) * np.sign(fi)
        else:
            arrow_end = mass_x + MASS_W + 0.01
        force_arrow.set_positions(
            (mass_x + MASS_W, 0),
            (arrow_end, 0)
        )
        force_text.set_position((max(mass_x + MASS_W, arrow_end) + 0.15, 0.35))
        force_text.set_text(f'F={fi:.1f} N')

        # Vreme
        time_text.set_text(f't = {ti:.3f} s')

        # Grafik
        line_x.set_data(t_sub[:i + 1], x_sub[:i + 1])
        line_v.set_data(t_sub[:i + 1], v_sub[:i + 1])
        cursor.set_data([ti, ti], [ax_plot.get_ylim()[0], ax_plot.get_ylim()[1]])

        return []

    update(0)

    anim = FuncAnimation(
        fig, update, frames=len(idx),
        interval=1000 // FPS, blit=False
    )

    plt.close(fig)

    print(f'Čuvanje animacije ({len(idx)} frejmova, {FPS} FPS)...')
    anim.save('animacija.gif', writer=PillowWriter(fps=FPS))
    print('Sačuvano: animacija.gif')


if __name__ == '__main__':
    create_animation()
