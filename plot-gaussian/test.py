# If you don't have these installed:
# pip install plotly ipywidgets

import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

DERIVATIVE_OPTIONS = [
    "Z (Gaussian)",
    "∂Z/∂x",
    "∂Z/∂y",
    "∂²Z/∂x²",
    "∂²Z/∂y²",
    "∂²Z/∂x∂y",
]

def compute_surface(X, Y, Z, sigma, derivative):
    """Return the selected derivative of the 2D Gaussian."""
    if derivative == "Z (Gaussian)":
        return Z
    elif derivative == "∂Z/∂x":
        return -(X / sigma**2) * Z
    elif derivative == "∂Z/∂y":
        return -(Y / sigma**2) * Z
    elif derivative == "∂²Z/∂x²":
        return ((X**2 - sigma**2) / sigma**4) * Z
    elif derivative == "∂²Z/∂y²":
        return ((Y**2 - sigma**2) / sigma**4) * Z
    elif derivative == "∂²Z/∂x∂y":
        return (X * Y / sigma**4) * Z
    return Z


def gaussian_dzdx_surface(
    sigma=4.0,
    extent_sigma=4.0,
    n=160,
    z_scale=1.0,
    derivative="∂Z/∂x",
    temp_gamma=1.0,
    temp_clip_max=6.0,
    color_intensity=1.0,
    color_cmin=0.0,
    color_cmax=1.0,
    colorscale="Inferno",
    show_sigma_rings=True,
    ring_sigmas=(1, 2, 3),
    grid_x=False,
    grid_y=False,
    grid_z=False,
    grid_x_size=20,
    grid_y_size=20,
    grid_z_size=20,
):
    lim = float(extent_sigma) * float(sigma)
    x = np.linspace(-lim, lim, int(n))
    y = np.linspace(-lim, lim, int(n))
    X, Y = np.meshgrid(x, y)

    Z = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))

    S = compute_surface(X, Y, Z, sigma, derivative)

    R = np.sqrt(X**2 + Y**2) / sigma
    Rc = np.clip(R, 0.0, float(temp_clip_max))
    Rn = Rc / float(temp_clip_max) if temp_clip_max > 0 else Rc
    Rg = np.power(Rn, float(temp_gamma)) if temp_gamma > 0 else Rn

    Rg_scaled = np.clip(Rg * float(color_intensity), 0.0, 1.0)

    contours_kwargs = {}
    if grid_x:
        contours_kwargs["x"] = dict(
            show=True, highlight=False,
            project=dict(x=True), size=(2 * lim) / grid_x_size,
            color="rgba(255,255,255,0.3)",
        )
    if grid_y:
        contours_kwargs["y"] = dict(
            show=True, highlight=False,
            project=dict(y=True), size=(2 * lim) / grid_y_size,
            color="rgba(255,255,255,0.3)",
        )
    if grid_z:
        zrange = float(np.ptp(S * z_scale))
        contours_kwargs["z"] = dict(
            show=True, highlight=False,
            project=dict(z=True), size=zrange / grid_z_size if zrange > 0 else 0.01,
            color="rgba(255,255,255,0.3)",
        )

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y, z=S * float(z_scale),
        surfacecolor=Rg_scaled,
        colorscale=colorscale,
        cmin=float(color_cmin), cmax=float(color_cmax),
        colorbar=dict(
            title="Deviation (σ), mapped",
            tickvals=[color_cmin, (color_cmin + color_cmax) / 2, color_cmax],
            ticktext=[
                f"{color_cmin * temp_clip_max:.2f}",
                f"{(color_cmin + color_cmax) / 2 * temp_clip_max:.2f}",
                f"{color_cmax * temp_clip_max:.2f}",
            ],
        ),
        contours=contours_kwargs if contours_kwargs else None,
        showscale=True,
        name=derivative,
    ))

    if show_sigma_rings:
        zmin = float((S * z_scale).min())
        z_ring = zmin - 0.05 * abs(zmin) if zmin != 0 else -1e-6

        for k in ring_sigmas:
            k = float(k)
            r = k * sigma
            t = np.linspace(0, 2 * np.pi, 400)
            xr = r * np.cos(t)
            yr = r * np.sin(t)
            zr = np.full_like(t, z_ring)
            fig.add_trace(go.Scatter3d(
                x=xr, y=yr, z=zr,
                mode="lines",
                line=dict(width=6),
                name=f"{k:.0f}σ ring",
                showlegend=True,
            ))

        if derivative in ("∂Z/∂x", "∂²Z/∂x²"):
            for sgn in (-1, 1):
                xl = np.full(400, sgn * sigma)
                yl = np.linspace(-lim, lim, 400)
                zl = np.full(400, z_ring)
                fig.add_trace(go.Scatter3d(
                    x=xl, y=yl, z=zl,
                    mode="lines",
                    line=dict(width=6),
                    name=f"x = {sgn:+.0f}σ",
                    showlegend=True,
                ))
        elif derivative in ("∂Z/∂y", "∂²Z/∂y²"):
            for sgn in (-1, 1):
                yl = np.full(400, sgn * sigma)
                xl = np.linspace(-lim, lim, 400)
                zl = np.full(400, z_ring)
                fig.add_trace(go.Scatter3d(
                    x=xl, y=yl, z=zl,
                    mode="lines",
                    line=dict(width=6),
                    name=f"y = {sgn:+.0f}σ",
                    showlegend=True,
                ))

    fig.update_layout(
        title=f"3D Surface: {derivative} of 2D Gaussian, σ = {sigma:.2f}",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title=derivative,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=720,
    )

    return fig

# --- Widgets ---

# Gaussian shape
sigma_w = widgets.FloatSlider(value=4.0, min=0.5, max=12.0, step=0.1,
                               description="sigma", continuous_update=False)
extent_w = widgets.FloatSlider(value=4.0, min=1.5, max=8.0, step=0.5,
                                description="extent(σ)", continuous_update=False)
n_w = widgets.IntSlider(value=160, min=60, max=260, step=20,
                         description="grid n", continuous_update=False)

# Derivative selector
deriv_w = widgets.Dropdown(
    options=DERIVATIVE_OPTIONS,
    value="∂Z/∂x",
    description="derivative",
)

# Z-scale
zscale_w = widgets.FloatLogSlider(value=1.0, base=10, min=-2, max=2, step=0.1,
                                   description="z_scale", continuous_update=False)

# Temperature / deviation mapping
gamma_w = widgets.FloatSlider(value=1.0, min=0.2, max=3.0, step=0.1,
                               description="temp_gamma", continuous_update=False)
clip_w = widgets.FloatSlider(value=6.0, min=1.0, max=12.0, step=0.5,
                              description="temp_clip", continuous_update=False)

# Color scaling intensity
intensity_w = widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1,
                                   description="intensity", continuous_update=False)
cmin_w = widgets.FloatSlider(value=0.0, min=0.0, max=0.9, step=0.05,
                              description="color min", continuous_update=False)
cmax_w = widgets.FloatSlider(value=1.0, min=0.1, max=1.0, step=0.05,
                              description="color max", continuous_update=False)

# Colormap
cmap_w = widgets.Dropdown(
    options=["Inferno", "Turbo", "Viridis", "Plasma", "Magma", "Cividis"],
    value="Inferno",
    description="colors",
)

# Sigma rings
rings_w = widgets.Checkbox(value=True, description="sigma rings")
ring_levels_w = widgets.SelectMultiple(
    options=[1, 2, 3, 4, 5],
    value=(1, 2, 3),
    description="ring σ",
)

# Overlay grid lines
grid_x_w = widgets.Checkbox(value=False, description="grid X")
grid_y_w = widgets.Checkbox(value=False, description="grid Y")
grid_z_w = widgets.Checkbox(value=False, description="grid Z")
grid_x_size_w = widgets.IntSlider(value=20, min=4, max=60, step=2,
                                   description="X lines", continuous_update=False)
grid_y_size_w = widgets.IntSlider(value=20, min=4, max=60, step=2,
                                   description="Y lines", continuous_update=False)
grid_z_size_w = widgets.IntSlider(value=20, min=4, max=60, step=2,
                                   description="Z lines", continuous_update=False)

out = widgets.Output()

def redraw(*_):
    with out:
        out.clear_output(wait=True)
        fig = gaussian_dzdx_surface(
            sigma=sigma_w.value,
            extent_sigma=extent_w.value,
            n=n_w.value,
            z_scale=zscale_w.value,
            derivative=deriv_w.value,
            temp_gamma=gamma_w.value,
            temp_clip_max=clip_w.value,
            color_intensity=intensity_w.value,
            color_cmin=cmin_w.value,
            color_cmax=cmax_w.value,
            colorscale=cmap_w.value,
            show_sigma_rings=rings_w.value,
            ring_sigmas=tuple(ring_levels_w.value),
            grid_x=grid_x_w.value,
            grid_y=grid_y_w.value,
            grid_z=grid_z_w.value,
            grid_x_size=grid_x_size_w.value,
            grid_y_size=grid_y_size_w.value,
            grid_z_size=grid_z_size_w.value,
        )
        fig.show()

all_widgets = [
    sigma_w, extent_w, n_w, deriv_w, zscale_w,
    gamma_w, clip_w, intensity_w, cmin_w, cmax_w,
    cmap_w, rings_w, ring_levels_w,
    grid_x_w, grid_y_w, grid_z_w,
    grid_x_size_w, grid_y_size_w, grid_z_size_w,
]
for w in all_widgets:
    w.observe(redraw, names="value")

controls = widgets.VBox([
    widgets.HTML("<b>Gaussian & Grid</b>"),
    widgets.HBox([sigma_w, extent_w, n_w]),
    widgets.HTML("<b>Derivative & Scale</b>"),
    widgets.HBox([deriv_w, zscale_w]),
    widgets.HTML("<b>Color Mapping</b>"),
    widgets.HBox([gamma_w, clip_w, intensity_w]),
    widgets.HBox([cmin_w, cmax_w, cmap_w]),
    widgets.HTML("<b>Overlay Grid Lines</b>"),
    widgets.HBox([grid_x_w, grid_y_w, grid_z_w]),
    widgets.HBox([grid_x_size_w, grid_y_size_w, grid_z_size_w]),
    widgets.HTML("<b>Sigma Rings</b>"),
    widgets.HBox([rings_w, ring_levels_w]),
])

display(controls, out)
redraw()
